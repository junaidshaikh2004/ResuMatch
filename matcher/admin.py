from django.contrib import admin

from .models import MatchResult, Resume


@admin.register(Resume)
class ResumeAdmin(admin.ModelAdmin):
    list_display = ("original_filename", "file_type", "session_key", "uploaded_at")
    readonly_fields = ("uploaded_at",)
    search_fields = ("session_key", "original_filename")


@admin.register(MatchResult)
class MatchResultAdmin(admin.ModelAdmin):
    list_display = ("verdict", "score", "primary_model", "used_stored_resume", "created_at")
    list_filter = ("verdict", "primary_model", "used_stored_resume")
    readonly_fields = ("created_at",)
