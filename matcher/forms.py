from django import forms

from django.conf import settings


class ResumeUploadForm(forms.Form):
    """Used both for the permanent 'My Resume' upload and the one-time upload choice."""

    resume_file = forms.FileField()

    def clean_resume_file(self):
        uploaded_file = self.cleaned_data["resume_file"]

        extension = "." + uploaded_file.name.lower().rsplit(".", 1)[-1] if "." in uploaded_file.name else ""
        if extension not in settings.ALLOWED_RESUME_EXTENSIONS:
            allowed = ", ".join(settings.ALLOWED_RESUME_EXTENSIONS)
            raise forms.ValidationError(f"Unsupported file type. Allowed types: {allowed}")

        max_bytes = settings.MAX_RESUME_UPLOAD_SIZE_MB * 1024 * 1024
        if uploaded_file.size > max_bytes:
            raise forms.ValidationError(f"File too large. Max size is {settings.MAX_RESUME_UPLOAD_SIZE_MB}MB.")

        return uploaded_file


RESUME_SOURCE_CHOICES = [
    ("stored", "Use my stored resume"),
    ("one_time", "Upload a different resume just for this check"),
]


class JobCheckForm(forms.Form):
    jd_text = forms.CharField(
        widget=forms.Textarea(attrs={
            "rows": 10,
            "placeholder": "Paste the job description here...",
            "class": "w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-indigo-500",
        }),
        label="Job description",
        min_length=30,
    )
    resume_source = forms.ChoiceField(
        choices=RESUME_SOURCE_CHOICES,
        widget=forms.RadioSelect,
        initial="stored",
    )
    one_time_resume_file = forms.FileField(required=False)

    def clean(self):
        cleaned_data = super().clean()
        source = cleaned_data.get("resume_source")
        one_time_file = cleaned_data.get("one_time_resume_file")

        if source == "one_time":
            if not one_time_file:
                raise forms.ValidationError("Please upload a resume file for this one-time check.")

            extension = "." + one_time_file.name.lower().rsplit(".", 1)[-1] if "." in one_time_file.name else ""
            if extension not in settings.ALLOWED_RESUME_EXTENSIONS:
                allowed = ", ".join(settings.ALLOWED_RESUME_EXTENSIONS)
                raise forms.ValidationError(f"Unsupported file type. Allowed types: {allowed}")

        return cleaned_data
