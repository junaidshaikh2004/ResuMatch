from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse

from matcher.models import MatchResult, Resume

SAMPLE_JD = (
    "We are hiring a Backend Developer with 3+ years of experience in Python, "
    "Django, and SQL. Docker experience is a plus."
)


def make_txt_file(name, content):
    return SimpleUploadedFile(name, content.encode("utf-8"), content_type="text/plain")


class ResumeUploadFlowTests(TestCase):
    def test_upload_creates_resume_for_session(self):
        response = self.client.post(
            reverse("my_resume"),
            {"resume_file": make_txt_file("resume.txt", "Backend developer skilled in Python and Django.")},
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(Resume.objects.count(), 1)

    def test_replacing_resume_updates_instead_of_duplicating(self):
        self.client.post(
            reverse("my_resume"),
            {"resume_file": make_txt_file("resume.txt", "First version of the resume.")},
        )
        self.client.post(
            reverse("my_resume"),
            {"resume_file": make_txt_file("resume_v2.txt", "Second, updated version of the resume.")},
        )
        self.assertEqual(Resume.objects.count(), 1)
        resume = Resume.objects.first()
        self.assertIn("Second", resume.raw_text)
        self.assertEqual(resume.original_filename, "resume_v2.txt")


class StoredVsOneTimeResumeTests(TestCase):
    def setUp(self):
        self.stored_text = "Stored resume: experienced Python and Django developer with 5 years experience."
        self.client.post(
            reverse("my_resume"),
            {"resume_file": make_txt_file("stored_resume.txt", self.stored_text)},
        )

    def test_one_time_upload_does_not_replace_stored_resume(self):
        response = self.client.post(
            reverse("job_check"),
            {
                "jd_text": SAMPLE_JD,
                "resume_source": "one_time",
                "one_time_resume_file": make_txt_file(
                    "one_time.txt", "A completely different one-time resume about marketing."
                ),
            },
        )
        self.assertEqual(response.status_code, 302)

        # The stored resume must be untouched.
        self.assertEqual(Resume.objects.count(), 1)
        self.assertEqual(Resume.objects.first().raw_text, self.stored_text)

        # The match result should reflect that a one-time resume was used.
        match_result = MatchResult.objects.latest("created_at")
        self.assertFalse(match_result.used_stored_resume)
        self.assertIn("marketing", match_result.resume_text_used.lower())

    def test_stored_resume_is_used_when_selected(self):
        response = self.client.post(
            reverse("job_check"),
            {"jd_text": SAMPLE_JD, "resume_source": "stored"},
        )
        self.assertEqual(response.status_code, 302)

        match_result = MatchResult.objects.latest("created_at")
        self.assertTrue(match_result.used_stored_resume)
        self.assertEqual(match_result.resume_text_used, self.stored_text)

    def test_stored_option_without_a_stored_resume_shows_error(self):
        # A fresh client has no stored resume at all.
        fresh_client = self.client_class()
        response = fresh_client.post(
            reverse("job_check"),
            {"jd_text": SAMPLE_JD, "resume_source": "stored"},
        )
        self.assertEqual(response.status_code, 200)  # re-renders the form with an error
        self.assertContains(response, "don&#x27;t have a stored resume")
