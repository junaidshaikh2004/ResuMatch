# ResuMatch

ResuMatch checks how well a resume fits a job description. Upload a resume once
(it's parsed and stored server-side, tied to your browser session — no login
needed), paste a job description, and get a fit score backed by a real NLP +
machine learning pipeline: TF-IDF, sentence embeddings, skill/experience
extraction, and two models — a Logistic Regression baseline and a hand-built
PyTorch neural network — trained from scratch and compared side by side on a
dashboard.

## How it works

1. **Resume storage** — a resume (PDF/DOCX/TXT) is parsed into plain text and
   stored in the database, keyed by Django session. The "My Resume" page lets
   you view or replace it at any time.
2. **Job fit check** — paste a job description, then choose to use your
   stored resume or upload a different one just for that one check (which is
   never saved).
3. **Feature engineering** (`matcher/ml/features.py`) turns a resume/JD pair
   into 7 numbers: TF-IDF cosine similarity, sentence-embedding cosine
   similarity, skill overlap ratio, missing-skill count, resume years of
   experience, JD years required, and the experience gap.
4. **Two models**, both trained on those same features
   (`matcher/management/commands/train_models.py`):
   - Logistic Regression (`class_weight="balanced"`) as an interpretable baseline.
   - `FitNet`, a small hand-built PyTorch feedforward network with a real
     manual training loop.
   Both are evaluated on a held-out test set (accuracy, precision, recall,
   F1, confusion matrix), and because missing a genuine good fit matters
   more than raw accuracy, each model's decision threshold is tuned on a
   separate validation set to maximize recall on the "Good Fit" class.
5. **Dashboard** (`/dashboard/`) shows dataset EDA (class balance, feature
   distributions) and both models' metrics side by side — the "which
   approach works better and why" story.

## About the dataset

No suitable public dataset of labeled resume↔job-description **fit** pairs
was reliably available (public options were either resume corpora labeled
only by job *category*, with no JD pairing or fit label, or personal-project
repos with unclear licensing/unusable structure). `data/resume_jd_dataset.csv`
is therefore an **AI-generated synthetic dataset**, built by
`matcher/management/commands/generate_dataset.py` from template sentences
and a curated skill bank across 12 job roles, with labels assigned by a
skill-overlap + experience rule plus a small amount of injected noise. It is
not real user data — this is disclosed here and at the top of that file.

## Project layout

```
resumatch/            Django project settings, root URLs, WSGI entry point
matcher/               the app: models, views, forms, templates
matcher/ml/            text extraction, preprocessing, feature engineering, PyTorch model, inference
matcher/management/commands/   generate_dataset.py, train_models.py
data/                  generated dataset (CSV)
ml_artifacts/          trained models, vectorizer, scaler, metrics/EDA JSON
```

## Local setup

```bash
python -m venv venv
venv\Scripts\activate          # on Windows
pip install -r requirements.txt

copy .env.example .env         # then edit values if you want to change defaults

python manage.py migrate
python manage.py generate_dataset   # builds data/resume_jd_dataset.csv
python manage.py train_models       # trains both models, writes ml_artifacts/
python manage.py test
python manage.py runserver
```

The dataset and trained artifacts are committed to the repo, so a fresh
clone works immediately after `pip install` + `migrate` without needing to
regenerate/retrain anything — the two commands above are for reproducing or
retraining them from scratch.

## Deploying to Render (free tier)

This app is deploy-ready but deployment itself is done manually, not by this
tool. Steps, for when you're ready:

1. Push the repo to GitHub.
2. Create a new **Web Service** on Render pointing at the repo.
3. **Build command**: `pip install -r requirements.txt && python manage.py collectstatic --noinput && python manage.py migrate`
4. **Start command**: `gunicorn resumatch.wsgi:application --workers 1 --threads 2 --timeout 120`
   (a `Procfile` with the same line is included for platforms that read one directly).
5. Add a free Render Postgres database and copy its **Internal Database URL**
   into the `DATABASE_URL` environment variable.
6. Set environment variables (see `.env.example`): `SECRET_KEY` (generate a
   real one), `DEBUG=False`, `ALLOWED_HOSTS=<your-app>.onrender.com`,
   `CSRF_TRUSTED_ORIGINS=https://<your-app>.onrender.com`.
7. Deploy.

**Known constraint**: Render's free tier gives 512MB RAM. `torch` and
`sentence-transformers` are heavier dependencies than a typical Django app —
this project mitigates that by using a small embedding model
(`all-MiniLM-L6-v2`, ~80MB), loading it once per process as a singleton
(`matcher/ml/predict.py`) instead of per-request, running a single gunicorn
worker, and committing the already-trained model files so the deployed app
never trains on boot. If you see out-of-memory errors, the single-worker
setting above is the first thing to check (don't scale up gunicorn workers
without also scaling up the plan's RAM).

## Tests

`python manage.py test` covers:
- Resume text extraction from PDF, DOCX, and plain text (`test_parsing.py`).
- The stored-vs-one-time-upload resume flow, including that a one-time
  upload never overwrites the stored resume (`test_resume_flow.py`).
- The end-to-end matching pipeline returning a valid score range and labels
  (`test_matching.py`).
