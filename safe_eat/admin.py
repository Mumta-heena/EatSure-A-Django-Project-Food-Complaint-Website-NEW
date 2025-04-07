from django.contrib import admin
from .models import *
# Register your models here.

@admin.register(ContactMessage)
class ContactMessageAdmin(admin.ModelAdmin):
    list_display = ('name', 'email', 'subject', 'created_at')
    list_filter = ('subject',)
    search_fields = ('name', 'email', 'subject', 'message')
    ordering = ('-created_at',)

admin.site.register([restaurant, complaint,PendingRestaurant,Profile])
