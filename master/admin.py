from django.contrib import admin
from .models import *
from django.contrib.auth.admin import UserAdmin
from django.utils.translation import gettext_lazy as _



class CustomUserAdmin(UserAdmin):
    fieldsets = (
        (None, {"fields": ("username", "password")}),
        (_("Personal info"), {"fields": ("first_name", "last_name", "email", "role")}),
        (
            _("Permissions"),
            {
                "fields": (
                    "is_active",
                    "is_staff",
                    "is_superuser",
                    "groups",
                    "user_permissions",
                    
                ),
            },
        ),
        (_("Important dates"), {"fields": ("last_login", "date_joined")}),
    )
    add_fieldsets = (
        (
            None,
            {
                "classes": ("wide",),
                "fields": ("username", "password1", "password2", "role"),
            },
        ),
    )

admin.site.register(User, CustomUserAdmin)



class LocationAdmin(admin.ModelAdmin):
    pass
    
admin.site.register(Location, LocationAdmin)




class DoctorAdmin(admin.ModelAdmin):
    readonly_fields = ('id',)
    
admin.site.register(Doctor, DoctorAdmin)


class NurseAdmin(admin.ModelAdmin):
    readonly_fields = ('id',)
    
admin.site.register(Nurse, NurseAdmin)


class PatientAdmin(admin.ModelAdmin):
    readonly_fields = ('id',)
    
admin.site.register(Patient, PatientAdmin)


class DepartmentAdmin(admin.ModelAdmin):
    readonly_fields = ('id',)
admin.site.register(Department, DepartmentAdmin)


class AppointmentAdmin(admin.ModelAdmin):
    readonly_fields = ('id',)
admin.site.register(Appointment, AppointmentAdmin)


class MedicineAdmin(admin.ModelAdmin):
    readonly_fields = ('id',)
admin.site.register(Medicine, MedicineAdmin)



class PatientDischargeDetailsAdmin(admin.ModelAdmin):
    readonly_fields = ('doctor_report',)
admin.site.register(PatientDischargeDetails, PatientDischargeDetailsAdmin)