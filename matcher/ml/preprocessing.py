"""
Light text cleaning shared by every part of the pipeline (TF-IDF, skill
matching, experience extraction). We deliberately keep this simple: lowercase,
collapse whitespace, and strip characters that aren't letters/digits/basic
punctuation. We do NOT stem or remove stopwords here — scikit-learn's
TfidfVectorizer handles stopword removal itself, and stemming would make
skill names (e.g. "React", "AWS") harder to match reliably.
"""

import re

_WHITESPACE_RE = re.compile(r"\s+")
_KEEP_CHARS_RE = re.compile(r"[^a-z0-9+#./\s-]")


def clean_text(text):
    text = text.lower()
    # Keep +, #, ., / and - since they show up in real skill names
    # (C++, C#, Node.js, CI/CD).
    text = _KEEP_CHARS_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()
