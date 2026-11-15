from django.contrib import admin

from .models import ChatSession, Lead, Message


class MessageInline(admin.TabularInline):
    model = Message
    extra = 0
    can_delete = False
    fields = ("created_at", "role", "content", "elapsed_ms", "sources")
    readonly_fields = fields

    def has_add_permission(self, request, obj=None):
        return False


@admin.register(Lead)
class LeadAdmin(admin.ModelAdmin):
    list_display = ("created_at", "full_name", "phone", "company", "status")
    list_filter = ("status", "created_at")
    list_editable = ("status",)
    search_fields = ("full_name", "phone", "company", "summary")
    readonly_fields = ("session", "created_at", "dialog")
    fields = ("status", "full_name", "phone", "company", "summary", "created_at", "session", "dialog")

    @admin.display(description="Диалог")
    def dialog(self, obj):
        return "\n".join(f"{m.get_role_display()}: {m.content}" for m in obj.session.messages.all())


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ("__str__", "updated_at", "message_count", "has_lead", "contact_name", "contact_company")
    list_filter = ("created_at",)
    search_fields = ("contact_name", "contact_phone", "contact_company", "messages__content")
    readonly_fields = ("id", "created_at", "updated_at")
    inlines = [MessageInline]

    @admin.display(description="Сообщений")
    def message_count(self, obj):
        return obj.messages.count()

    @admin.display(description="Заявка", boolean=True)
    def has_lead(self, obj):
        return hasattr(obj, "lead")
