from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("resume/", views.my_resume, name="my_resume"),
    path("check/", views.job_check, name="job_check"),
    path("check/result/<int:pk>/", views.result, name="result"),
    path("dashboard/", views.dashboard, name="dashboard"),
]
