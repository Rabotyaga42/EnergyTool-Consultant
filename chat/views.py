import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.clickjacking import xframe_options_exempt
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_POST

from . import service

SESSION_KEY = "chat_session_id"


def _current_session(request):
    chat = service.get_or_create_session(request.session.get(SESSION_KEY))
    request.session[SESSION_KEY] = str(chat.id)
    return chat


@xframe_options_exempt  # чат можно встроить в сайт Energytool через iframe
@ensure_csrf_cookie
def index(request):
    chat = _current_session(request)
    return render(request, "chat/index.html", {"history": service.history(chat)})


@require_POST
def send(request):
    try:
        text = json.loads(request.body).get("message", "")
    except (ValueError, AttributeError):
        return JsonResponse({"error": "Некорректный запрос"}, status=400)
    try:
        return JsonResponse(service.handle_user_message(_current_session(request), text))
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)


@require_POST
def new_dialog(request):
    request.session[SESSION_KEY] = str(service.new_session().id)
    return JsonResponse({"ok": True})
