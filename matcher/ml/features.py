"""
Turns a (resume_text, jd_text) pair into the numeric feature vector that both
models are trained on, plus a few extra human-readable bits (matched/missing
skills) used only for display.

The same functions here are called from train_models.py (to build the
training set) and predict.py (to score a live request), so the features a
model was trained on are exactly the features it sees at inference time.
"""

import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import clean_text
from .skills import ALL_SKILLS

# Fixed order of the numeric feature vector fed to both models. Both
# train_models.py and predict.py must build vectors in this exact order.
FEATURE_NAMES = [
    "tfidf_similarity",
    "skill_overlap_ratio",
    "missing_skill_count",
    "resume_years",
    "jd_years_required",
    "experience_gap",
]

_YEARS_RE = re.compile(r"(\d+)\+?\s*(?:years|yrs|year)\b")


def extract_years_of_experience(text):
    """
    Pulls out the largest "N years" mention in the text, e.g. "5+ years of
    experience" or "3 years". Returns 0.0 if nothing is found — treated as
    "no experience requirement/claim stated" rather than missing data, which
    keeps the feature numeric and simple.
    """
    matches = _YEARS_RE.findall(clean_text(text))
    if not matches:
        return 0.0
    return float(max(int(m) for m in matches))


def extract_skills(text):
    """Returns the set of known skills (from skills.ALL_SKILLS) mentioned in text."""
    cleaned = clean_text(text)
    found = set()
    for skill in ALL_SKILLS:
        # Word-boundary match so "react" doesn't match inside "reaction".
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, cleaned):
            found.add(skill)
    return found


def tfidf_similarity(resume_text, jd_text, vectorizer):
    """Cosine similarity between resume/JD TF-IDF vectors using an already-fitted vectorizer."""
    vectors = vectorizer.transform([clean_text(resume_text), clean_text(jd_text)])
    return float(cosine_similarity(vectors[0], vectors[1])[0][0])


def build_feature_dict(resume_text, jd_text, vectorizer):
    """
    Computes every feature plus the matched/missing skill lists. Returns a
    plain dict so callers can pick out `FEATURE_NAMES` for the model input
    and use the rest (matched_skills, missing_skills) for display.
    """
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    matched_skills = sorted(resume_skills & jd_skills)
    missing_skills = sorted(jd_skills - resume_skills)
    skill_overlap_ratio = len(matched_skills) / len(jd_skills) if jd_skills else 0.0

    resume_years = extract_years_of_experience(resume_text)
    jd_years_required = extract_years_of_experience(jd_text)

    return {
        "tfidf_similarity": tfidf_similarity(resume_text, jd_text, vectorizer),
        "skill_overlap_ratio": skill_overlap_ratio,
        "missing_skill_count": float(len(missing_skills)),
        "resume_years": resume_years,
        "jd_years_required": jd_years_required,
        "experience_gap": resume_years - jd_years_required,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
    }


def feature_dict_to_vector(feature_dict):
    """Extracts the ordered numeric vector (FEATURE_NAMES) a model expects."""
    return np.array([feature_dict[name] for name in FEATURE_NAMES], dtype=np.float64)
