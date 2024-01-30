from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser


class CustomUserAdmin(UserAdmin):
    model = CustomUser
    list_display = ['username', 'email', 'organization', 'is_staff']
    fieldsets = UserAdmin.fieldsets + (
        (None, {'fields': ('organization', 'profile_picture', 'bio')}),
    )
    add_fieldsets = UserAdmin.add_fieldsets + (
        (None, {'fields': ('organization', 'profile_picture', 'bio')}),
    )


admin.site.register(CustomUser, CustomUserAdmin)
