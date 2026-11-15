import uuid

from django.db import models


class ChatSession(models.Model):
    """Один диалог. «Новый диалог» создаёт новую сессию, старая остаётся в базе."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField("Начало", auto_now_add=True)
    updated_at = models.DateTimeField("Последняя активность", auto_now=True)
    contact_name = models.CharField("ФИО", max_length=200, blank=True)
    contact_phone = models.CharField("Телефон", max_length=50, blank=True)
    contact_company = models.CharField("Компания", max_length=200, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Диалог"
        verbose_name_plural = "Диалоги"

    def __str__(self):
        return f"Диалог {str(self.id)[:8]} от {self.created_at:%d.%m.%Y %H:%M}"


class Message(models.Model):
    USER = "user"
    ASSISTANT = "assistant"
    ROLES = [(USER, "Клиент"), (ASSISTANT, "Консультант")]

    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name="messages")
    role = models.CharField("Роль", max_length=10, choices=ROLES)
    content = models.TextField("Текст")
    sources = models.JSONField("Документы базы знаний", default=list, blank=True)
    elapsed_ms = models.PositiveIntegerField("Время ответа, мс", null=True, blank=True)
    created_at = models.DateTimeField("Время", auto_now_add=True)

    class Meta:
        ordering = ["id"]
        verbose_name = "Сообщение"
        verbose_name_plural = "Сообщения"

    def __str__(self):
        return f"{self.get_role_display()}: {self.content[:50]}"


class Lead(models.Model):
    NEW, IN_WORK, DONE = "new", "in_work", "done"
    STATUSES = [(NEW, "Новая"), (IN_WORK, "В работе"), (DONE, "Обработана")]

    session = models.OneToOneField(ChatSession, on_delete=models.CASCADE, related_name="lead")
    full_name = models.CharField("ФИО", max_length=200)
    phone = models.CharField("Телефон", max_length=50)
    company = models.CharField("Компания", max_length=200)
    summary = models.TextField("Резюме диалога")
    status = models.CharField("Статус", max_length=10, choices=STATUSES, default=NEW)
    created_at = models.DateTimeField("Создана", auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Заявка"
        verbose_name_plural = "Заявки"

    def __str__(self):
        return f"{self.full_name} ({self.company})"
