import os
import sys

from django.apps import AppConfig


class ChatConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "chat"
    verbose_name = "Консультант Energytool"

    def ready(self):
        # Прогрев индекса базы знаний в фоне: gunicorn или runserver (основной процесс автоперезапуска).
        served = "gunicorn" in sys.argv[0] or ("runserver" in sys.argv and os.environ.get("RUN_MAIN") == "true")
        if served and os.environ.get("WARMUP", "1") == "1":
            from . import service
            service.warmup()
