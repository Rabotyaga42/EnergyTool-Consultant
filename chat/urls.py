from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/chat/", views.send, name="send"),
    path("api/new/", views.new_dialog, name="new_dialog"),
]
