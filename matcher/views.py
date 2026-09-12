import json

from django.contrib import messages
from django.shortcuts import get_object_or_404, redirect, render
from django.conf import settings

from .forms import JobCheckForm, ResumeUploadForm
from .ml import predict as predict_module
from .ml.text_extraction import EmptyResumeText, UnsupportedFileType, extract_text
from .models import MatchResult, Resume
from .session_utils import get_or_create_session_key


def home(request):
    session_key = get_or_create_session_key(request)
    has_resume = Resume.objects.filter(session_key=session_key).exists()
    return render(request, "matcher/home.html", {"has_resume": has_resume})


def my_resume(request):
    session_key = get_or_create_session_key(request)
    resume = Resume.objects.filter(session_key=session_key).first()

    if request.method == "POST":
        form = ResumeUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.cleaned_data["resume_file"]
            try:
                text, file_type = extract_text(uploaded_file, uploaded_file.name)
            except (UnsupportedFileType, EmptyResumeText) as exc:
                form.add_error("resume_file", str(exc))
            else:
                resume, _created = Resume.objects.update_or_create(
                    session_key=session_key,
                    defaults={
                        "original_filename": uploaded_file.name,
                        "file_type": file_type,
                        "raw_text": text,
                    },
                )
                messages.success(request, "Your resume has been saved.")
                return redirect("my_resume")
    else:
        form = ResumeUploadForm()

    return render(request, "matcher/my_resume.html", {"resume": resume, "form": form})


def job_check(request):
    session_key = get_or_create_session_key(request)
    has_resume = Resume.objects.filter(session_key=session_key).exists()

    if request.method == "POST":
        form = JobCheckForm(request.POST, request.FILES)
        if form.is_valid():
            source = form.cleaned_data["resume_source"]
            jd_text = form.cleaned_data["jd_text"]

            if source == "stored":
                resume = Resume.objects.filter(session_key=session_key).first()
                if not resume:
                    form.add_error(None, "You don't have a stored resume yet. Upload one below, or visit My Resume first.")
                    return render(request, "matcher/job_check.html", {"form": form, "has_resume": has_resume})
                resume_text = resume.raw_text
                used_stored = True
            else:
                uploaded_file = form.cleaned_data["one_time_resume_file"]
                try:
                    resume_text, _file_type = extract_text(uploaded_file, uploaded_file.name)
                except (UnsupportedFileType, EmptyResumeText) as exc:
                    form.add_error("one_time_resume_file", str(exc))
                    return render(request, "matcher/job_check.html", {"form": form, "has_resume": has_resume})
                used_stored = False

            result = predict_module.predict(resume_text, jd_text)

            match_result = MatchResult.objects.create(
                session_key=session_key,
                jd_text=jd_text,
                resume_text_used=resume_text,
                used_stored_resume=used_stored,
                score=result["score"],
                verdict=result["verdict"],
                primary_model=result["primary_model"],
                other_model_score=result["other_model_score"],
                other_model_verdict=result["other_model_verdict"],
                other_model_name=result["other_model_name"],
                matched_skills=result["matched_skills"],
                missing_skills=result["missing_skills"],
                resume_years_experience=result["resume_years"],
                jd_years_experience_required=result["jd_years_required"],
            )
            return redirect("result", pk=match_result.pk)
    else:
        form = JobCheckForm()

    return render(request, "matcher/job_check.html", {"form": form, "has_resume": has_resume})


def result(request, pk):
    session_key = get_or_create_session_key(request)
    match_result = get_object_or_404(MatchResult, pk=pk, session_key=session_key)
    context = {
        "result": match_result,
        # match_result.score/.other_model_score are 0-1 probabilities;
        # convert to whole-number percentages here rather than in the
        # template, since Django templates can't do arithmetic cleanly.
        "score_percent": round(match_result.score * 100),
        "other_score_percent": round(match_result.other_model_score * 100),
    }
    return render(request, "matcher/result.html", context)


def dashboard(request):
    with open(settings.ML_ARTIFACTS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    with open(settings.ML_ARTIFACTS_DIR / "eda.json") as f:
        eda = json.load(f)

    return render(request, "matcher/dashboard.html", {"metrics": metrics, "eda": eda})
