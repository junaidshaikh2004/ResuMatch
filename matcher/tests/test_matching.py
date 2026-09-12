from django.test import SimpleTestCase

from matcher.ml import predict

RESUME_TEXT = (
    "Backend developer with 5 years of experience in Python, Django, SQL, "
    "and Docker. Comfortable working independently or in a team."
)
JD_TEXT = (
    "We are hiring a Backend Developer with 3+ years of experience in "
    "Python, Django, PostgreSQL, and Docker."
)


class PredictPipelineTests(SimpleTestCase):
    """
    Runs the full feature-engineering + model pipeline end to end. Requires
    trained artifacts to exist (run `python manage.py generate_dataset` then
    `python manage.py train_models` before running tests).
    """

    def test_predict_returns_sane_output(self):
        result = predict.predict(RESUME_TEXT, JD_TEXT)

        self.assertIsInstance(result["score"], float)
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1.0)

        self.assertIn(result["verdict"], {"Good Fit", "Not a Fit"})
        self.assertIn(result["other_model_verdict"], {"Good Fit", "Not a Fit"})
        self.assertIn(result["primary_model"], {"logistic_regression", "pytorch_nn"})

        self.assertIsInstance(result["matched_skills"], list)
        self.assertIsInstance(result["missing_skills"], list)
        # This pair overlaps heavily on Python/Django/Docker.
        self.assertIn("python", result["matched_skills"])
        self.assertIn("django", result["matched_skills"])

    def test_unrelated_resume_scores_lower_than_matching_one(self):
        unrelated_resume = (
            "Graphic designer with 4 years of experience in Adobe Photoshop, "
            "branding, and print design."
        )
        matching_result = predict.predict(RESUME_TEXT, JD_TEXT)
        unrelated_result = predict.predict(unrelated_resume, JD_TEXT)

        self.assertGreater(matching_result["score"], unrelated_result["score"])
