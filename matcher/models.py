from django.db import models


class Resume(models.Model):
    """
    The one resume a visitor has stored, keyed by their Django session.
    There's no login system, so the session cookie is what ties a resume to
    "whose" it is. Storing the extracted plain text (not the original file)
    means the app survives a restart on free-tier hosting without needing
    persistent file storage.
    """

    session_key = models.CharField(max_length=40, unique=True)
    original_filename = models.CharField(max_length=255)
    file_type = models.CharField(max_length=10)  # "pdf", "docx", or "txt"
    raw_text = models.TextField()
    uploaded_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Resume({self.original_filename}) for session {self.session_key[:8]}..."


class MatchResult(models.Model):
    """
    A saved record of one job-fit check. Persisting this (instead of just
    passing the result through the session) lets the result page use a
    plain POST-redirect-GET flow and gives the admin something real to
    inspect.
    """

    session_key = models.CharField(max_length=40)
    jd_text = models.TextField()
    resume_text_used = models.TextField()
    used_stored_resume = models.BooleanField(default=True)

    # Primary model's output (the one shown as "the" verdict).
    score = models.FloatField()
    verdict = models.CharField(max_length=20)  # "Good Fit" or "Not a Fit"
    primary_model = models.CharField(max_length=30)

    # The other model's output, shown as a smaller comparison line.
    other_model_score = models.FloatField()
    other_model_verdict = models.CharField(max_length=20)
    other_model_name = models.CharField(max_length=30)

    matched_skills = models.JSONField(default=list)
    missing_skills = models.JSONField(default=list)
    resume_years_experience = models.FloatField(null=True)
    jd_years_experience_required = models.FloatField(null=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"MatchResult({self.verdict}, {self.score:.2f}) at {self.created_at:%Y-%m-%d %H:%M}"
