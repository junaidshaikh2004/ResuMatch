import io
from pathlib import Path

from django.test import SimpleTestCase

from matcher.ml.text_extraction import (
    EmptyResumeText,
    UnsupportedFileType,
    extract_text,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TextExtractionTests(SimpleTestCase):
    def test_extracts_text_from_pdf_fixture(self):
        with open(FIXTURES_DIR / "sample_resume.pdf", "rb") as f:
            text, file_type = extract_text(f, "sample_resume.pdf")
        self.assertEqual(file_type, "pdf")
        self.assertIn("python", text.lower())

    def test_extracts_text_from_docx_fixture(self):
        with open(FIXTURES_DIR / "sample_resume.docx", "rb") as f:
            text, file_type = extract_text(f, "sample_resume.docx")
        self.assertEqual(file_type, "docx")
        self.assertIn("django", text.lower())

    def test_extracts_text_from_plain_txt(self):
        file_obj = io.BytesIO(b"Experienced backend developer skilled in Django and SQL.")
        text, file_type = extract_text(file_obj, "resume.txt")
        self.assertEqual(file_type, "txt")
        self.assertIn("Django", text)

    def test_unsupported_extension_raises(self):
        file_obj = io.BytesIO(b"not a real resume")
        with self.assertRaises(UnsupportedFileType):
            extract_text(file_obj, "resume.exe")

    def test_empty_text_raises(self):
        file_obj = io.BytesIO(b"   \n\t  ")
        with self.assertRaises(EmptyResumeText):
            extract_text(file_obj, "resume.txt")
