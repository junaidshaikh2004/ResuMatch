"""
Turns an uploaded resume file (PDF, DOCX, or plain text) into plain text.

Each format needs a different library, so this module is just a small
dispatcher plus one function per format. Nothing fancy: we don't try to
preserve layout or sections, just pull out readable text, which is all the
downstream NLP pipeline needs.
"""

import docx
import pdfplumber


class UnsupportedFileType(Exception):
    pass


class EmptyResumeText(Exception):
    pass


def extract_text_from_pdf(file_obj):
    text_parts = []
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_text_from_docx(file_obj):
    document = docx.Document(file_obj)
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def extract_text_from_txt(file_obj):
    raw = file_obj.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    return raw


def extract_text(file_obj, filename):
    """
    Dispatches on the file extension and returns extracted plain text.
    Raises UnsupportedFileType or EmptyResumeText for bad input, which the
    view turns into a friendly form error.
    """
    extension = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if extension == "pdf":
        text = extract_text_from_pdf(file_obj)
    elif extension == "docx":
        text = extract_text_from_docx(file_obj)
    elif extension == "txt":
        text = extract_text_from_txt(file_obj)
    else:
        raise UnsupportedFileType(f"Unsupported file type: .{extension}")

    text = text.strip()
    if not text:
        raise EmptyResumeText("No readable text was found in this file.")

    return text, extension
