import django_filters
from .models import *


class DepartmentFilter(django_filters.FilterSet):
    class  Meta:
        model = Department
        fields = {'name':{'icontains'}, }
        
        
class MedicineFilter(django_filters.FilterSet):
    class  Meta:
        model = Medicine
        fields = {'name':{'exact'}, 'price':{'lt','gt'}, 'department':{'exact'}}


class DoctorFilter(django_filters.FilterSet):
    class  Meta:
        model = Doctor
        fields = {'user__is_active':{'exact'}, 'user__first_name':{'icontains'}, 'department':{'exact'}}


class NurseFilter(django_filters.FilterSet):
    class  Meta:
        model = Nurse
        fields = {'user__is_active':{'exact'}, 'user__first_name':{'icontains'}, 'department':{'exact'}}

      

class PatientFilter(django_filters.FilterSet):
    class  Meta:
        model = Patient
        fields = {'user__is_active':{'exact'}, 'user__first_name':{'icontains'}, 'assigned_doctor':{'exact'}}



class AppointmentFilter(django_filters.FilterSet):
    class  Meta:
        model = Appointment
        fields = {'appointment_date':{'exact'}, 'appointment_time':{'exact'}}

