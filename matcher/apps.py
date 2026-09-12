import logging
import os
import sys

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class MatcherConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "matcher"

    def ready(self):
        if not self._should_warm_ml_artifacts():
            return

        from .ml import predict

        try:
            predict.warm()
        except Exception:
            # Don't crash app startup over this — e.g. on a fresh clone
            # before `train_models` has been run yet. predict() will still
            # try (and log) again lazily on the first real request.
            logger.exception("Could not pre-load ML artifacts at startup")

    def _should_warm_ml_artifacts(self):
        """
        Only worth eagerly loading the model artifacts when we're actually
        about to serve requests: under gunicorn (production), or
        `manage.py runserver` (local dev). Skip it for every other manage.py
        command (migrate, test, generate_dataset, train_models, shell, ...)
        so those stay fast and don't require artifacts to already exist.
        """
        argv0 = sys.argv[0] if sys.argv else ""
        is_manage_py = "manage.py" in argv0

        if not is_manage_py:
            return True  # e.g. gunicorn importing resumatch.wsgi

        if len(sys.argv) < 2 or sys.argv[1] != "runserver":
            return False

        # With the autoreloader on (the default), Django re-executes this
        # module in a parent watcher process before spawning the real child
        # process, which sets RUN_MAIN — only that child should warm. With
        # --noreload there's no such split: this process IS the real one.
        return "--noreload" in sys.argv or os.environ.get("RUN_MAIN") == "true"
