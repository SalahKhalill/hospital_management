from django import forms
from .models import *
from .widget import CustomImageFieldWidget




        
        
class LocationForm(forms.ModelForm):
    class Meta:
        model = Location
        fields = "__all__"

#for user signup
class UserForm(forms.ModelForm):
    username = forms.CharField(disabled=True)
    class Meta:
        model=User
        fields=('username', 'first_name','last_name', 'email')
       

class DoctorForm(forms.ModelForm):
    profile_pic = forms.ImageField(widget=CustomImageFieldWidget)
    department_name = forms.CharField(disabled=True, required=False, label="Department")
    specialization = forms.CharField(disabled=True, required=False)
    class Meta:
        model=Doctor
        fields= ("profile_pic", "department_name", "specialization")
    
    def __init__(self, *args, **kwargs):
        super(DoctorForm, self).__init__(*args, **kwargs)
        if self.instance.department:
            self.fields['department_name'].initial = self.instance.department.name
        

        
class AdminDoctorForm(forms.ModelForm):
    username = forms.CharField(disabled=True)
    first_name = forms.CharField(disabled=True,required=False)
    last_name = forms.CharField(disabled=True,required=False)
    address = forms.CharField(disabled=True,required=False)
    email = forms.CharField(disabled=True,required=False)
    class Meta:
        model=Doctor
        fields= ('username', 'first_name','last_name', "specialization", "department", "address", "email") 
        
    def __init__(self, *args, **kwargs):
        super(AdminDoctorForm, self).__init__(*args, **kwargs)
        if self.instance.user.username:
            self.fields['username'].initial = self.instance.user.username
        if self.instance.user.first_name:
            self.fields['first_name'].initial = self.instance.user.first_name
        if self.instance.user.last_name:
            self.fields['last_name'].initial = self.instance.user.last_name
        if self.instance.user.email:
            self.fields['email'].initial = self.instance.user.email
        if self.instance.location.governorate:
            self.fields['address'].initial = self.instance.location.governorate
            



class NurseForm(forms.ModelForm):
    profile_pic = forms.ImageField(widget=CustomImageFieldWidget)
    department_name = forms.CharField(disabled=True, required=False, label="Department")
    shift_time = forms.CharField(disabled=True, required=False)
    class Meta:
        model=Nurse
        fields= ("profile_pic", "department_name", "shift_time")
    
    def __init__(self, *args, **kwargs):
        super(NurseForm, self).__init__(*args, **kwargs)
        if self.instance.department:
            self.fields['department_name'].initial = self.instance.department.name
    

class AdminNurseForm(forms.ModelForm):
    username = forms.CharField(disabled=True)
    first_name = forms.CharField(disabled=True,required=False)
    last_name = forms.CharField(disabled=True,required=False)
    address = forms.CharField(disabled=True,required=False)
    email = forms.CharField(disabled=True,required=False)
    class Meta:
        model=Nurse
        fields= ('username', 'first_name','last_name', "shift_time", "department", "address", "email") 
        
    def __init__(self, *args, **kwargs):
        super(AdminNurseForm, self).__init__(*args, **kwargs)
        if self.instance.user.username:
            self.fields['username'].initial = self.instance.user.username
        if self.instance.user.first_name:
            self.fields['first_name'].initial = self.instance.user.first_name
        if self.instance.user.last_name:
            self.fields['last_name'].initial = self.instance.user.last_name
        if self.instance.user.email:
            self.fields['email'].initial = self.instance.user.email
        if self.instance.location.governorate:
            self.fields['address'].initial = self.instance.location.governorate
    


class PatientForm(forms.ModelForm):
    #this is the extrafield for linking patient and their assigend doctor
    #this will show dropdown __str__ method doctor model is shown on html so override it
    #to_field_name this will fetch corresponding value  user_id present in Doctor model and return it
    profile_pic = forms.ImageField(widget=CustomImageFieldWidget)
    class Meta:
        model=Patient
        fields=("profile_pic", "symptoms", "assigned_doctor")


class AppointmentForm(forms.ModelForm):
    class Meta:
        model=Appointment
        fields=('patient', 'department', 'doctor', 'appointment_date', 'appointment_time','description')



#for departments page
class DepartmentForm(forms.ModelForm):
    department_pic = forms.ImageField(widget=CustomImageFieldWidget)
    class Meta:
        model = Department
        fields= "__all__"




#for medicines page
class MedicineForm(forms.ModelForm):
    class Meta:
        model=Medicine
        fields= "__all__"
        
        
class PatientDischargeForm(forms.ModelForm):
        doctor_report = forms.CharField(widget=forms.Textarea(attrs={'rows': 4, 'placeholder': 'Enter doctor report or diagnosis details...'}))  
        class Meta:
            model = PatientDischargeDetails
            fields = ('doctor_report','room_charge', 'medicine_cost', 'doctor_fee', 'other_charge',)


class DoctorPatientDischargeForm(forms.ModelForm):
    class Meta:
        model = PatientDischargeDetails
        fields = ('doctor_report',)


class ContactForm(forms.ModelForm):
    username = forms.CharField(disabled=True)
    first_name = forms.CharField(disabled=True,required=False)
    last_name = forms.CharField(disabled=True,required=False)
    email = forms.CharField(disabled=True)
    message = forms.CharField(widget=forms.Textarea(attrs={'rows': 4}))
    class Meta:
        model=User
        fields=('username', 'first_name','last_name', 'email')

