import os
from django.utils.http import urlencode
from django.conf import settings
from xhtml2pdf import pisa
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm, UsernameField, PasswordChangeForm
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.utils.decorators import method_decorator
from django.contrib import messages
from django.views import View
from django.template.loader import get_template
from.forms import *
from .filters import *
from django.core.mail import send_mail
from .ai_classifier import (
    classify_image, ClassifierType, ClassificationResult,
    ModelNotFoundError, ClassificationError, ImageValidationError, ImageQualityError,
    check_models_status, get_classifier_info, Severity, ImageQuality,
    get_all_conditions, get_model_info
)
from PIL import Image
import io
import base64
import tempfile
import logging

logger = logging.getLogger(__name__)



#custom the UserCreationForm to update the auth user in it
class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(required=True, help_text='Required. Enter a valid email address.')
    
    class Meta:
        model = User
        fields = ("username", "first_name", "last_name", "email")
        field_classes = {"username": UsernameField}



# Create your views here.
def home_view(request):
    departments = Department.objects.all()
    return render(request, 'new_home.html', {'departments':departments})


#-----------for checking user is doctor , patient or admin(by submit)
def is_admin(user):
    return True if user.role =='ADMIN' else False
def is_doctor(user):
    return True if user.role =='DOCTOR' else False
def is_nurse(user):
    return True if user.role =='NURSE' else False
def is_patient(user):
    return True if user.role =='PATIENT' else False


def login_view(request, hospital_user):
    if request.method == "POST":
        login_form = AuthenticationForm(request=request, data=request.POST)
        if login_form.is_valid():
            username = login_form.cleaned_data.get('username')
            password = login_form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                # Check if account is approved (skip check for superusers)
                if not user.is_approved:
                    if not user.is_superuser:
                        messages.warning(request, f"Your account is not approved yet. Please wait for approval or contact admin.")
                        return redirect('home')

                # Check if password change is required
                if user.password_change_required:
                    login(request, user)
                    return redirect('force_password_change')

                if is_admin(user):
                    if user.is_staff:
                        login(request, user)
                        messages.success(request, f"You are logged successfully as {user.username}.")
                        return redirect('after_login')
                    else:
                        messages.warning(request, f"Your account is not approved yet. Please wait for approval or contact admin.")
                        return redirect('home')
                else:
                    if user.is_active:
                        login(request, user)
                        messages.success(request, f"You are logged successfully as {user.username}.")
                        return redirect('after_login')
                    else:
                        messages.warning(request, f"Your account is not approved yet. Please wait for approval or contact admin.")
                        return redirect('home')
            else:
                messages.warning(request, f"Invalid username or password.")
        else:
            messages.warning(request, f"Invalid username or password.")
    elif request.method == "GET":
        login_form = AuthenticationForm()
    return render(request, 'new_login.html', {"login_form": login_form, "hospital_user": hospital_user})



@login_required
def logout_view(request):
    logout(request)
    messages.warning(request, f"You are logged out and you are now without an account")
    return redirect('home')


def register_view(request):
    if request.method == 'POST':
        register_form = CustomUserCreationForm(data=request.POST)
        if register_form.is_valid():
            user = register_form.save(commit=False)
            user.role = 'PATIENT'  # Always create as patient
            user.is_active = False
            user.save()
            user.refresh_from_db()
            messages.success(request, f"Your data has been registered successfully. You can log in as {user.username} after admin approval.")
            return redirect("home")
        else:
            messages.error(request, f"An error occured try again.")
    elif request.method == 'GET':
        register_form = CustomUserCreationForm()
    return render(request, 'new_register.html', {'register_form':register_form, 'hospital_user':'patient'})







#---------AFTER ENTERING CREDENTIALS WE CHECK WHETHER USERNAME AND PASSWORD IS OF ADMIN,DOCTOR OR PATIENT
@login_required
def after_login_view(request):
    if is_admin(request.user):
        return redirect('admin_dashboard')
    elif is_doctor(request.user):
        return redirect('doctor_dashboard')
    elif is_nurse(request.user):
        return redirect('nurse_dashboard')
    elif is_patient(request.user):
        return redirect('patient_dashboard')




@method_decorator(login_required, name='dispatch')
@method_decorator(user_passes_test(is_admin), name='dispatch')
class AdminProfileView(View):
    def get(self, request):
        user_form = UserForm(instance=request.user)
        return render(request, 'after_login/new_admin_profile.html', {'user_form':user_form})
    
    def post(self, request):
        user_form = UserForm(request.POST, instance=request.user)
        if user_form.is_valid():
            user_form.save()
            messages.success(request, 'Profile Updated')
            return redirect("admin_profile")
        else:
            messages.warning(request, 'Error Updating Profile')
        return render(request, 'after_login/new_admin_profile.html', {'user_form':user_form})
        




@method_decorator(login_required, name='dispatch')
class UserProfileView(View):
    def get(self, request):
        user_form = UserForm(instance=request.user)
        profile_form = None
        location_form = None
        
        if is_doctor(request.user):
            profile_form = DoctorForm(instance=request.user.doctor)
            location_form = LocationForm(instance=request.user.doctor.location)
        elif is_nurse(request.user):
            profile_form = NurseForm(instance=request.user.nurse)
            location_form = LocationForm(instance=request.user.nurse.location)
        elif is_patient(request.user):
            profile_form = PatientForm(instance=request.user.patient)
            location_form = LocationForm(instance=request.user.patient.location)
        
        return render(request, 'after_login/new_user_profile.html',
                      {'user_form':user_form, 'profile_form':profile_form, 'location_form':location_form})
    
    def post(self, request):
        user_form = UserForm(request.POST, instance=request.user)
        profile_form = None
        location_form = None
        
        if is_doctor(request.user):
            profile_form = DoctorForm(request.POST, request.FILES, instance=request.user.doctor)
            location_form = LocationForm(request.POST, instance=request.user.doctor.location)
        elif is_nurse(request.user):
            profile_form = NurseForm(request.POST, request.FILES, instance=request.user.nurse)
            location_form = LocationForm(request.POST, instance=request.user.nurse.location)
        elif is_patient(request.user):
            profile_form = PatientForm(request.POST, request.FILES, instance=request.user.patient)
            location_form = LocationForm(request.POST, instance=request.user.patient.location)
        
        # For admin users who only have user_form
        if profile_form is None or location_form is None:
            if user_form.is_valid():
                user_form.save()
                messages.success(request, 'Profile Updated')
                return redirect("user_profile")
            else:
                messages.warning(request, 'Error Updating Profile')
        elif user_form.is_valid() and profile_form.is_valid() and location_form.is_valid():
            user_form.save()
            profile_form.save()
            location_form.save()
            messages.success(request, 'Profile Updated')
            return redirect("user_profile")
        else:
            messages.warning(request, 'Error Updating Profile')
        return render(request, 'after_login/new_user_profile.html',
                      {'user_form':user_form, 'profile_form':profile_form, 'location_form':location_form})
        

@login_required
def change_password_view(request):
    if request.method == 'POST':
        password_form = PasswordChangeForm(user=request.user, data=request.POST)
        if password_form.is_valid():
            password_form.save()
            messages.success(request, f"Password Is Updated Successfully")
            if is_admin(request.user):
                return redirect("admin_profile")
            else:
                return redirect("user_profile")
        else:
            messages.warning(request, 'Error Updating Password Try Again')
    else:
        password_form = PasswordChangeForm(user=request.user)
    return render(request, 'after_login/new_change_password.html', {"password_form": password_form})


@login_required
def force_password_change_view(request):
    """Force users to change their password on first login"""
    if not request.user.password_change_required:
        return redirect('after_login')
    
    if request.method == 'POST':
        password_form = PasswordChangeForm(user=request.user, data=request.POST)
        if password_form.is_valid():
            user = password_form.save()
            user.password_change_required = False
            user.temp_password = None
            user.save()
            update_session_auth_hash(request, user)  # Keep user logged in
            messages.success(request, "Password changed successfully! You can now access your account.")
            return redirect('after_login')
        else:
            messages.error(request, 'Error changing password. Please try again.')
    else:
        password_form = PasswordChangeForm(user=request.user)
        # Show temporary password if available
        temp_pass_msg = f" Your temporary password was: {request.user.temp_password}" if request.user.temp_password else ""
    
    return render(request, 'after_login/force_password_change.html', {
        'password_form': password_form,
        'temp_password': request.user.temp_password
    })





#---------------------------------------------------------------------------------
#------------------------ ADMIN RELATED VIEWS START ------------------------------
#---------------------------------------------------------------------------------
@login_required()
@user_passes_test(is_admin)
def admin_dashboard_view(request):
    #for both table in admin dashboard
    doctors=Doctor.objects.all().order_by('-id')
    departments=Department.objects.all().order_by('-id')
    medicines=Medicine.objects.all().order_by('-id')
    nurses=Nurse.objects.all().order_by('-id')
    patients=Patient.objects.all().order_by('-id')
    #for three cards
    doctorcount=Doctor.objects.all().filter(user__is_active=True).count()
    pendingdoctorcount=Doctor.objects.all().filter(user__is_active=False).count()
    
    departmentcount=Department.objects.all().count()
    medicinecount=Medicine.objects.all().count()

    
    nursecount=Nurse.objects.all().filter(user__is_active=True).count()
    pendingnursecount=Nurse.objects.all().filter(user__is_active=False).count()

    patientcount=Patient.objects.all().filter(user__is_active=True).count()
    pendingpatientcount=Patient.objects.all().filter(user__is_active=False).count()

    appointmentcount=Appointment.objects.all().filter(status=True, is_done=False).count()
    pendingappointmentcount=Appointment.objects.all().filter(status=False).count()
    
    mydict={
    'doctors':doctors,
    'medicines':medicines,
    'departments':departments,
    'nurses':nurses,
    'patients':patients,
    'doctorcount':doctorcount,
    'pendingdoctorcount':pendingdoctorcount,
    'departmentcount':departmentcount,
    'medicinecount':medicinecount,
    'nursecount':nursecount,
    'pendingnursecount':pendingnursecount,
    'patientcount':patientcount,
    'pendingpatientcount':pendingpatientcount,
    'appointmentcount':appointmentcount,
    'pendingappointmentcount':pendingappointmentcount,
    }
    return render(request,'after_login/new_admin_dashboard.html',context=mydict)




@login_required
@user_passes_test(is_admin)
def admin_manager_view(request, data):
    if data == 'MEDICINE':
        medicines = Medicine.objects.all().order_by('-id')
        data_table = MedicineFilter(request.GET, medicines)
    elif data == 'DEPARTMENT':
        departments = Department.objects.all().order_by('-id')
        data_table = DepartmentFilter(request.GET, departments)
    elif data == 'DOCTOR':
        doctors = Doctor.objects.all().order_by('-id')
        data_table = DoctorFilter(request.GET, doctors)
    elif data == 'NURSE':
        nurses = Nurse.objects.all().order_by('-id')
        data_table = NurseFilter(request.GET, nurses)
    elif data == 'PATIENT':
        patients = Patient.objects.all().order_by('-id')
        data_table = PatientFilter(request.GET, patients)
    elif data == 'APPOINTMENT':
        appointments = Appointment.objects.all().order_by('-id')
        data_table = AppointmentFilter(request.GET, appointments)
        
    context = {'data_table': data_table, 'data_name' : data}
    return render(request, 'after_login/new_admin_manager.html', context)



@login_required
@user_passes_test(is_admin)
def admin_confirm_user_view(request, id):
    user=User.objects.get(id=id)
    role = user.role
    user.is_active = True
    user.save()
    messages.success(request, f"The user {user.username} has been successfully confirmed as a {role}.")
    return redirect('admin_manager', data=role)


@login_required
@user_passes_test(is_admin)
def admin_delete_user_view(request, id):
    user = User.objects.get(id=id)
    role = user.role
    
    try:
        # If deleting a doctor, first unassign them from all patients
        if role == 'DOCTOR' and hasattr(user, 'doctor'):
            Patient.objects.filter(assigned_doctor=user.doctor).update(assigned_doctor=None)
        
        user.delete()
        messages.warning(request, f"The user has been deleted from the hospital database.")
    except Exception as e:
        messages.error(request, f"Cannot delete this user. They may have related records (appointments, etc.) that need to be deleted first.")
    
    return redirect('admin_manager', data=role)



@login_required
@user_passes_test(is_admin)
def admin_delete_department_view(request, id):
    department=Department.objects.get(id=id)
    department.delete()
    messages.warning(request, f"The department has been deleted from the hospital database.")
    return redirect('admin_manager', data='DEPARTMENT')


@login_required
@user_passes_test(is_admin)
def admin_delete_medicine_view(request, id):
    medicine=Medicine.objects.get(id=id)
    medicine.delete()
    messages.warning(request, f"The medication has been deleted from the hospital database.")
    return redirect('admin_manager', data='MEDICINE')



@login_required
@user_passes_test(is_admin)
def admin_create_staff_view(request):
    """Admin creates staff (doctors, nurses) with auto-generated passwords"""
    import random
    import string
    
    if request.method == 'POST':
        role = request.POST.get('role')
        username = request.POST.get('username')
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        email = request.POST.get('email')
        initial_password = request.POST.get('initial_password', '').strip()
        
        if not all([role, username, first_name, last_name, email]):
            messages.error(request, "All fields are required.")
            return render(request, 'after_login/new_admin_create_staff.html', {'roles': User.Role.choices})
        
        # Check if username already exists
        if User.objects.filter(username=username).exists():
            messages.error(request, f"Username '{username}' already exists. Please choose another.")
            return render(request, 'after_login/new_admin_create_staff.html', {'roles': User.Role.choices})
        
        # Use provided password or generate random one
        if initial_password:
            if len(initial_password) < 8:
                messages.error(request, "Password must be at least 8 characters long.")
                return render(request, 'after_login/new_admin_create_staff.html', {'roles': User.Role.choices})
            temp_password = initial_password
        else:
            temp_password = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        
        # Create user
        user = User.objects.create_user(
            username=username,
            password=temp_password,
            first_name=first_name,
            last_name=last_name,
            email=email,
            role=role
        )
        user.is_active = True
        user.is_approved = True
        user.approved_by = request.user
        user.approved_at = timezone.now()
        user.password_change_required = True
        user.temp_password = temp_password
        
        if role == 'ADMIN':
            user.is_staff = True
        
        user.save()
        
        # Send email with credentials
        try:
            send_mail(
                subject='Your Account Has Been Created - MedCare Hospital',
                message=f'''Dear {user.get_full_name()},

Your account has been created successfully!

Username: {username}
Temporary Password: {temp_password}
Role: {user.get_role_display()}

Important: You MUST change your password on first login.

Please login at: http://localhost:8000/{role.lower()}/login/

Best regards,
MedCare Hospital Administration''',
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[email],
                fail_silently=True,
            )
        except:
            pass
        
        messages.success(request, f'''
            Account created successfully!<br>
            <strong>Username:</strong> {username}<br>
            <strong>Temporary Password:</strong> <code>{temp_password}</code><br>
            <strong>Role:</strong> {user.get_role_display()}<br>
            <em>User must change password on first login. Credentials sent to {email}.</em>
        ''')
        return redirect('admin_create_staff')
    
    roles = [(role[0], role[1]) for role in User.Role.choices if role[0] != 'PATIENT']
    return render(request, 'after_login/new_admin_create_staff.html', {'roles': roles})



@login_required
@user_passes_test(is_admin)
def admin_add_user_view(request, hospital_user):
    if request.method == 'POST':
        user_form = CustomUserCreationForm(data=request.POST)
        if user_form.is_valid():
            user = user_form.save(commit=False)
            user.role = hospital_user
            if hospital_user=='ADMIN':
                user.is_staff = True
                user.is_active = True
                user.is_approved = True
                user.approved_by = request.user
                user.approved_at = timezone.now()
                user.save()
                user.refresh_from_db()
                messages.success(request, f"The user {user.username} has been successfully added as a {hospital_user}.")
                return redirect('admin_dashboard')
            else:
                user.is_active = True
                user.is_approved = True
                user.approved_by = request.user
                user.approved_at = timezone.now()
                user.save()
                user.refresh_from_db()
                messages.success(request, f"The user {user.username} has been successfully added as a {hospital_user}.")
                return redirect('admin_manager', data=hospital_user)
        else:
            messages.error(request, f"An error occured try again.")
    elif request.method == 'GET':
        user_form = CustomUserCreationForm()
    return render(request, 'after_login/new_admin_add_user.html', {'user_form':user_form, 'hospital_user':hospital_user})



@login_required
@user_passes_test(is_admin)
def admin_pending_accounts_view(request):
    """View all pending accounts awaiting admin approval"""
    pending_users = User.objects.filter(is_approved=False, is_active=False).order_by('-created_at')
    
    # Separate by role for better organization
    pending_doctors = pending_users.filter(role='DOCTOR')
    pending_nurses = pending_users.filter(role='NURSE')
    pending_patients = pending_users.filter(role='PATIENT')
    
    context = {
        'pending_users': pending_users,
        'pending_doctors': pending_doctors,
        'pending_nurses': pending_nurses,
        'pending_patients': pending_patients,
        'total_pending': pending_users.count(),
    }
    return render(request, 'after_login/new_admin_pending_accounts.html', context)


@login_required
@user_passes_test(is_admin)
def admin_account_details_view(request, user_id):
    """View detailed information about a pending account"""
    user = get_object_or_404(User, id=user_id)
    
    # Get role-specific details
    profile = None
    location = None
    
    if is_doctor(user):
        profile = user.doctor
        location = user.doctor.location
    elif is_nurse(user):
        profile = user.nurse
        location = user.nurse.location
    elif is_patient(user):
        profile = user.patient
        location = user.patient.location
    
    context = {
        'account_user': user,
        'profile': profile,
        'location': location,
    }
    return render(request, 'after_login/new_admin_account_details.html', context)


@login_required
@user_passes_test(is_admin)
def admin_approve_account_view(request, user_id):
    """Approve a pending account"""
    user = get_object_or_404(User, id=user_id)
    
    if request.method == 'POST':
        user.is_approved = True
        user.is_active = True
        user.approved_by = request.user
        user.approved_at = timezone.now()
        user.save()
        
        messages.success(request, f"Account for {user.get_full_name() or user.username} ({user.role}) has been approved successfully.")
        
        # Send email notification if email exists
        if user.email:
            try:
                send_mail(
                    subject='Account Approved - Hospital Management System',
                    message=f'Dear {user.get_full_name() or user.username},\n\nYour account has been approved by the administrator. You can now log in to the system.\n\nUsername: {user.username}\n\nThank you.',
                    from_email=None,  # Will use DEFAULT_FROM_EMAIL from settings
                    recipient_list=[user.email],
                    fail_silently=True,
                )
            except Exception as e:
                logger.warning(f"Failed to send approval email to {user.email}: {str(e)}")
        
        return redirect('admin_pending_accounts')
    
    return redirect('admin_account_details', user_id=user_id)


@login_required
@user_passes_test(is_admin)
def admin_reject_account_view(request, user_id):
    """Reject and delete a pending account"""
    user = get_object_or_404(User, id=user_id)
    
    if request.method == 'POST':
        username = user.username
        role = user.role
        user.delete()
        
        messages.warning(request, f"Account for {username} ({role}) has been rejected and deleted.")
        return redirect('admin_pending_accounts')
    
    return redirect('admin_account_details', user_id=user_id)



@method_decorator(login_required, name='dispatch')
@method_decorator(user_passes_test(is_admin), name='dispatch')
class AdminUpdateUserView(View):
    def get(self, request, id):
        user=User.objects.get(id=id)
        form = None
        if is_doctor(user):
            form = AdminDoctorForm(instance=user.doctor)
        elif is_nurse(user):
            form = AdminNurseForm(instance=user.nurse)
        elif is_patient(user):
            form = PatientForm(instance=user.patient)
        return render(request, 'after_login/new_admin_update_user.html',{'form':form, 'hospital_user': user})
    
    def post(self, request, id):
        user=User.objects.get(id=id)
        form = None
        if is_doctor(user):
            form = AdminDoctorForm(request.POST, instance=user.doctor)
            if form.is_valid():
                # Handle password change if provided
                new_password = form.cleaned_data.get('new_password')
                if new_password:
                    user.set_password(new_password)
                    user.save()
                    messages.success(request, f"Password updated for {user.username}")
                form.save()
        elif is_nurse(user):
            form = AdminNurseForm(request.POST, instance=user.nurse)
            if form.is_valid():
                # Handle password change if provided
                new_password = form.cleaned_data.get('new_password')
                if new_password:
                    user.set_password(new_password)
                    user.save()
                    messages.success(request, f"Password updated for {user.username}")
                form.save()
        elif is_patient(user):
            form = PatientForm(request.POST, request.FILES, instance=user.patient)
            if form.is_valid():
                # Handle password change if provided
                new_password = form.cleaned_data.get('new_password')
                if new_password:
                    user.set_password(new_password)
                    user.save()
                    messages.success(request, f"Password updated for {user.username}")
                form.save()
        
        if form and form.is_valid():
            form.save()
            messages.success(request, f'{user.role} Updated')
            return redirect("admin_update_user", id=id)
        else:
            messages.warning(request, f'Error Updating {user.role}')
        return render(request, 'after_login/new_admin_update_user.html',{'form':form, 'hospital_user': user})
        




@login_required
@user_passes_test(is_admin)
def admin_add_medicine_view(request):
    if request.method == 'POST':
        medicine_form = MedicineForm(request.POST)
        if medicine_form.is_valid():
            medicine = medicine_form.save()
            messages.success(request, f"The medicine {medicine.name} has been added successfully.")
            return redirect('admin_manager', data='MEDICINE')
        else:
            messages.warning(request, f"An error occured try again.")
    elif request.method == 'GET':
        medicine_form = MedicineForm()
    return render(request, 'after_login/new_admin_medicine.html', {'medicine_form':medicine_form})




@login_required
@user_passes_test(is_admin)
def admin_update_medicine_view(request, id):
    if request.method == 'POST':
        medicine_form = MedicineForm(request.POST, instance=Medicine.objects.get(id=id))
        if medicine_form.is_valid():
            medicine = medicine_form.save()
            messages.success(request, f"The medicine {medicine.name} has been updated successfully.")
            return redirect('admin_manager', data='MEDICINE')
        else:
            messages.warning(request, f"An error occured try again.")
    elif request.method == 'GET':
        medicine_form = MedicineForm(instance=Medicine.objects.get(id=id))
    return render(request, 'after_login/new_admin_medicine.html', {'medicine_form':medicine_form})


@login_required
@user_passes_test(is_admin)
def admin_add_department_view(request):
    if request.method == 'POST':
        department_form = DepartmentForm(request.POST, request.FILES)
        if department_form.is_valid():
            department = department_form.save()
            messages.success(request, f"The department {department.name} has been added successfully.")
            return redirect('admin_manager', data='DEPARTMENT')
        else:
            messages.warning(request, f"An error occured try again.")
    elif request.method == 'GET':
        department_form = DepartmentForm()
    return render(request, 'after_login/new_admin_department.html', {'department_form':department_form})




@login_required
@user_passes_test(is_admin)
def admin_update_department_view(request, id):
    if request.method == 'POST':
        department_form = DepartmentForm(request.POST, request.FILES, instance=Department.objects.get(id=id))
        if department_form.is_valid():
            department = department_form.save()
            messages.success(request, f"The department {department.name} has been updated successfully.")
            return redirect('admin_manager', data='DEPARTMENT')
        else:
            messages.warning(request, f"An error occured try again.")
    elif request.method == 'GET':
        department_form = DepartmentForm(instance=Department.objects.get(id=id))
    return render(request, 'after_login/new_admin_department.html', {'department_form':department_form})


@login_required
@user_passes_test(is_admin)
def admin_add_appointment_view(request):
    if request.method == 'POST':
        appointment_form = AppointmentForm(request.POST)
        if appointment_form.is_valid():
            appointment = appointment_form.save(commit=False)
            appointment.status = True
            appointment.save()
            messages.success(request, f"The appointment has been successfully.")
            return redirect('admin_manager', data='APPOINTMENT')
        else:
            messages.warning(request, f"An error occured try again.")
    elif request.method == 'GET':
        appointment_form = AppointmentForm()
    return render(request, 'after_login/new_admin_appointment.html', {'appointment_form':appointment_form})



@login_required
@user_passes_test(is_admin)
def admin_confirm_appointment_view(request, id):
    appointment=Appointment.objects.get(id=id)
    appointment.status = True
    appointment.save()
    messages.success(request, f"The appointment has been successfully confirmed.")
    return redirect('admin_manager', data='APPOINTMENT')



@login_required
@user_passes_test(is_admin)
def admin_delete_appointment_view(request, id):
    appointment=Appointment.objects.get(id=id)
    appointment.delete()
    messages.warning(request, f"The appointment has been deleted from the hospital database.")
    return redirect('admin_manager', data='APPOINTMENT')



@login_required
@user_passes_test(is_admin)
def admin_discharge_patient_view(request, id):
    appointment = Appointment.objects.get(id=id)
    # Get or create discharge details for this appointment
    discharge_details, created = PatientDischargeDetails.objects.get_or_create(
        appointment=appointment,
        defaults={
            'doctor_report': '',
            'room_charge': 0,
            'medicine_cost': 0,
            'doctor_fee': 0,
            'other_charge': 0,
        }
    )
    
    if request.method == 'POST':
        discharge_form = PatientDischargeForm(request.POST, instance=discharge_details)
        if discharge_form.is_valid():
            form = discharge_form.save(commit=False)
            form.discharge_status = True
            form.save()
            appointment.is_done = True
            appointment.save()
            messages.success(request, f'The exit permit was successfully issued')
            return redirect('final_discharge', id=id)
        else:
            messages.error(request, f"An error occured try again.")
    else:
        discharge_form = PatientDischargeForm(instance=discharge_details)
    return render(request, 'after_login/new_admin_discharge_patient.html', {'discharge_form': discharge_form})




@login_required
def final_discharge_view(request, id):
    appointment = Appointment.objects.get(id= id)
    discharge_details = PatientDischargeDetails.objects.get(appointment= appointment)
    return render(request, 'after_login/new_final_discharge.html', {'discharge_form': discharge_details})


@login_required
def pay_discharge_view(request, id):
    discharge_form = PatientDischargeDetails.objects.get(id= id)
    discharge_form.is_paid = True
    discharge_form.save()
    messages.success(request, f'The permit costs have been paid successfully')
    if is_admin(request.user):
        return redirect('admin_manager', data='APPOINTMENT')
    elif is_patient(request.user):
        return redirect("patient_appointment")
    


def render_to_pdf(template_src, context_dict):
    template = get_template(template_src)
    html = template.render(context_dict)
    result = io.BytesIO()
    # تحويل صفحة HTML إلى ملف PDF باستخدام pisa
    pdf = pisa.pisaDocument(io.BytesIO(html.encode("UTF-8")), result, encoding="UTF-8")
    if not pdf.err:
        return result.getvalue()
    return None


@login_required
def download_permit_pdf(request, id):
    discharge_details = PatientDischargeDetails.objects.get(id=id)
    context = {'discharge_form': discharge_details}
    pdf = render_to_pdf('after_login/download_bill.html', context)

    if pdf:
        response = HttpResponse(pdf, content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="permit.pdf"'
        # response.write(pdf_content)
        return response

    # If there is an error generating the PDF, you can handle it here
    return HttpResponse('Error generating PDF', status=500)



def department_view(request, id):
    try:
        department = Department.objects.get(id=id)
        if department is None:
            raise Exception
        return render(request, 'services/new_department.html', {'department':department})
    except Exception as e:
        messages.warning(request, f'Invalid {id} id')
        return redirect('home')
    
    



#---------------------------------------------------------------------------------
#------------------------ DOCTOR RELATED VIEWS START ------------------------------
#---------------------------------------------------------------------------------
@login_required
@user_passes_test(is_doctor)
def doctor_dashboard_view(request):
    department = request.user.doctor.department
    nurses = Nurse.objects.filter(user__is_active=True, department=department)
    patients = Patient.objects.filter(user__is_active=True, assigned_doctor=request.user.doctor)
    
    nurse_count = nurses.count()
    patient_count = patients.count()
    appointment_count = Appointment.objects.filter(
        status=True, doctor=request.user.doctor, appointment_date=timezone.now().date()).count()
    
    context = {
        'department':department,
        'nurses': nurses,
        'patients': patients,
        'nurse_count': nurse_count,
        'patient_count': patient_count,
        'appointment_count': appointment_count,
    }
    
    return render(request, 'after_login/new_doctor_dashboard.html', context)



@login_required
@user_passes_test(is_doctor)
def doctor_appointment_view(request):
    appointments = Appointment.objects.filter(status=True, is_ready=False, doctor=request.user.doctor)
    appointment_data = AppointmentFilter(request.GET, appointments)
    return render(request, 'after_login/new_doctor_appointment.html', {'appointments': appointment_data})



@login_required
@user_passes_test(is_doctor)
def doctor_report_view(request, id):
    appointment = Appointment.objects.get(id= id)
    if request.method == 'POST':
        discharge_form = DoctorPatientDischargeForm(request.POST)
        if discharge_form.is_valid():
            form = discharge_form.save(commit=False)
            form.appointment = appointment
            form.save()
            appointment.is_ready=True
            appointment.save()
            messages.success(request, f'The report has been sent successfully')
            return redirect('doctor_appointment')
        else:
            messages.error(request, f"An error occured try again.")
    elif request.method == 'GET':
        discharge_form = DoctorPatientDischargeForm()
    return render(request, 'after_login/new_doctor_report.html', {'discharge_form': discharge_form})



#---------------------------------------------------------------------------------
#------------------------ NURSE RELATED VIEWS START ------------------------------
#---------------------------------------------------------------------------------
@login_required
@user_passes_test(is_nurse)
def nurse_dashboard_view(request):
    department = request.user.nurse.department
    nurses = Nurse.objects.filter(user__is_active=True, department=department)
    doctors = Doctor.objects.filter(user__is_active=True, department=department)
    appointments = Appointment.objects.filter(status=True, department=department, appointment_date=timezone.now().date())
    
    nurse_count = nurses.count()
    doctor_count = doctors.count()
    appointment_count = appointments.count()
    
    context = {
        'department': department,
        'nurses': nurses,
        'doctors': doctors,
        'nurse_count': nurse_count,
        'doctor_count': doctor_count,
        'appointment_count': appointment_count,
    }
    
    return render(request, 'after_login/new_nurse_dashboard.html', context)



@login_required
@user_passes_test(is_nurse)
def nurse_appointment_view(request):
    department = request.user.nurse.department
    appointments = Appointment.objects.filter(status=True, department=department)
    appointment_data = AppointmentFilter(request.GET, appointments)
    return render(request, 'after_login/new_nurse_appointment.html', {'appointments': appointment_data})





#---------------------------------------------------------------------------------
#------------------------ PATIENT RELATED VIEWS START ------------------------------
#---------------------------------------------------------------------------------
@login_required
@user_passes_test(is_patient)
def patient_dashboard_view(request):
    doctor = request.user.patient.assigned_doctor
    
    appointment_count = Appointment.objects.filter(
        status=True, patient=request.user.patient, appointment_date=timezone.now().date()).count()
    past_appointment_count = Appointment.objects.filter(
        status=True, patient=request.user.patient, appointment_date__lt=timezone.now().date()).count()
    future_appointment_count = Appointment.objects.filter(
        status=True, patient=request.user.patient, appointment_date__gt=timezone.now().date()).count()
    
    context = {
        'doctor': doctor,
        'appointment_count': appointment_count,
        'past_appointment_count': past_appointment_count,
        'future_appointment_count': future_appointment_count,
    }
    
    return render(request, 'after_login/new_patient_dashboard.html', context)



@login_required
@user_passes_test(is_patient)
def patient_appointment_view(request):
    appointments = Appointment.objects.filter(patient=request.user.patient)
    appointment_data = AppointmentFilter(request.GET, appointments)
    return render(request, 'after_login/new_patient_appointment.html', {'appointments': appointment_data})



@login_required
@user_passes_test(is_patient)
def patient_update_appointment_view(request, id):
    if request.method == 'POST':
        appointment_form = AppointmentForm(request.POST, instance=Appointment.objects.get(id=id))
        if appointment_form.is_valid():
            appointment = appointment_form.save(commit=False)
            appointment.status = False
            appointment.save()
            messages.success(request, f"The appointment has been updated successfully and will be confirmed.")
            return redirect('patient_appointment')
        else:
            messages.error(request, f"An error occured try again.")
    elif request.method == 'GET':
        appointment_form = AppointmentForm(instance=Appointment.objects.get(id=id))
    return render(request, 'after_login/new_book_appointment.html', {'appointment_form':appointment_form})



@login_required
@user_passes_test(is_patient)
def patient_delete_appointment_view(request, id):
    appointment=Appointment.objects.get(id=id)
    appointment.delete()
    messages.warning(request, f"The appointment has been deleted.")
    return redirect('patient_appointment')




@login_required
def book_appointment_view(request):
    if request.method == 'POST':
        appointment_form = AppointmentForm(request.POST)
        if appointment_form.is_valid():
            appointment = appointment_form.save()
            messages.success(request, f"The reservation has been added successfully and will be confirmed.")
            if is_doctor(request.user):
                return redirect('doctor_dashboard')
            elif is_nurse(request.user):
                return redirect('nurse_dashboard')
            elif is_patient(request.user):
                return redirect('patient_dashboard')
        else:
            messages.error(request, f"An error occured try again.")
    elif request.method == 'GET':
        appointment_form = AppointmentForm()
    return render(request, 'after_login/new_book_appointment.html', {'appointment_form':appointment_form})
    



def medicine_view(request):
    medicines = Medicine.objects.all().order_by('name')
    medicine_filter = MedicineFilter(request.GET, medicines)
    return render(request, 'services/new_medicine.html', {'medicines':medicine_filter})


def aboutus_view(request):
    departments = Department.objects.all().order_by('name')
    return render(request, 'services/new_aboutus.html', {'departments':departments})



@login_required
def contactus_view(request):
    if request.method == 'POST':
        form = ContactForm(request.POST, instance=request.user)
        if form.is_valid():
            message = form.cleaned_data['message']
            email_subject = f"Contact Form Submission from {request.user.username}"
            email_message = f"Name: {request.user.first_name} {request.user.last_name}\nEmail: {request.user.email}\nMessage: {message}"
            try:
                send_mail(email_subject, email_message, request.user.username, [settings.EMAIL_HOST_USER], fail_silently=True)
                messages.success(request, 'Your message has been sent!')
            except Exception as e:
                print(e)
                messages.error(request, 'An error occurred. Please try again.')
        else:
            messages.error(request, 'An error occurred. Please try again.')
    else:
        form = ContactForm(instance=request.user)
    return render(request, 'after_login/new_contactus.html', {'form':form})




@login_required
def send_mail_view(request, id):
    receiver = User.objects.get(id= id)
    if request.method == 'POST':
        form = ContactForm(request.POST, instance=request.user)
        if form.is_valid():
            message = form.cleaned_data['message']
            email_subject = f"Contact Form Submission from {request.user.role}/ {request.user.username}"
            email_message = f"Name: {request.user.first_name} {request.user.last_name}\nEmail: {request.user.email}\nMessage: {message}"
            try:
                send_mail(email_subject, email_message, request.user.username, [receiver.email], fail_silently=True)
                messages.success(request, 'Your message has been sent!')
            except Exception as e:
                print(e)
                messages.error(request, 'An error occurred. Please try again.')
        else:
            messages.error(request, 'An error occurred. Please try again.')
    else:
        form = ContactForm(instance=request.user)
    return render(request, 'after_login/new_contactus.html', {'form':form})




#---------------------------------------------------------------------------------
#------------------------ MEDICAL CLASSIFICATION RELATED VIEWS START ------------------------------
#---------------------------------------------------------------------------------

def _get_severity_color(severity: Severity) -> str:
    """Get Bootstrap color class for severity level."""
    colors = {
        Severity.NORMAL: 'success',
        Severity.LOW: 'info',
        Severity.MODERATE: 'warning',
        Severity.HIGH: 'danger',
        Severity.CRITICAL: 'danger',
    }
    return colors.get(severity, 'secondary')


def _process_ai_classification(request, classifier_type: ClassifierType, template_name: str):
    """
    Advanced medical classification handler with comprehensive results.
    
    Features:
    - Image preprocessing and enhancement
    - Image quality assessment
    - Test-time augmentation for robust predictions
    - Confidence scores with thresholds
    - Severity assessment
    - Top-N predictions with detailed information
    - Medical recommendations
    - Grad-CAM heatmap visualization
    - Performance metrics
    
    Args:
        request: Django HTTP request
        classifier_type: Type of classifier to use
        template_name: Template to render results
        
    Returns:
        Rendered template with classification results or error
    """
    classifier_info = get_classifier_info(classifier_type)
    model_info = get_model_info(classifier_type)
    all_conditions = get_all_conditions(classifier_type)
    
    context = {
        'classifier_info': classifier_info,
        'model_info': model_info,
        'all_conditions': all_conditions,
    }
    
    if request.method == 'POST' and request.FILES.get('image'):
        image_path = None
        try:
            # Load uploaded image
            image = request.FILES['image']
            img = Image.open(io.BytesIO(image.read()))
            
            # Store original size for display
            original_size = img.size
            
            # Validate image format
            if img.mode not in ('RGB', 'L', 'RGBA'):
                img = img.convert('RGB')
            
            # Create temporary file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                image_path = tmp_file.name
                img.save(image_path, format='PNG')
            
            # Get user options from form
            generate_heatmap = request.POST.get('generate_heatmap') == 'on'
            use_tta = request.POST.get('use_tta') == 'on'
            assess_quality = request.POST.get('assess_quality', 'on') == 'on'
            
            # Perform advanced classification with all features
            result = classify_image(
                image_path, 
                classifier_type,
                generate_heatmap=generate_heatmap,
                top_n=5,
                use_tta=use_tta,
                assess_quality=assess_quality
            )
            
            # Convert original image to base64 for display
            img = Image.open(image_path)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=90)
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Build comprehensive context with all results
            context.update({
                'result': result.class_name,
                'confidence': f"{result.confidence * 100:.1f}%",
                'confidence_value': result.confidence,
                'is_confident': result.is_confident,
                'severity': result.severity.value,
                'severity_color': _get_severity_color(result.severity),
                'urgency': result.urgency_level,
                'description': result.medical_info.get('description', ''),
                'top_predictions': [
                    {
                        'class': pred.class_name,
                        'probability': f"{pred.probability * 100:.1f}%",
                        'probability_value': pred.probability,
                        'description': pred.description,
                        'severity': pred.severity.value if pred.severity else None,
                    }
                    for pred in result.top_predictions
                ],
                'recommendations': result.recommendations,
                'preprocessing_applied': result.preprocessing_applied,
                'img': img_str,
                'original_size': f"{original_size[0]}x{original_size[1]}",
                'heatmap': result.heatmap_base64,
                'show_results': True,
                'inference_time': f"{result.inference_time_ms:.0f}",
                'used_tta': use_tta,
                'image_hash': result.image_hash[:12] if result.image_hash else None,
            })
            
            # Add quality assessment if available
            if result.quality_assessment:
                qa = result.quality_assessment
                # Calculate overall quality score from component scores
                avg_quality = (qa.brightness_score + qa.contrast_score + qa.sharpness_score + qa.noise_score) / 4
                context.update({
                    'quality_assessment': {
                        'overall_quality': qa.overall_quality.value,
                        'quality_score': f"{avg_quality * 100:.0f}%",
                        'brightness_score': f"{qa.brightness_score * 100:.0f}%",
                    }
                })
            
        except ModelNotFoundError as e:
            logger.error(f"Model not found for {classifier_type.value}: {str(e)}")
            context['error'] = (
                f"Classification model not available. Please ensure the {classifier_type.value}.h5 "
                "model file is downloaded and placed in the 'models' folder."
            )
            
        except ImageQualityError as e:
            logger.warning(f"Image quality issue for {classifier_type.value}: {str(e)}")
            context['error'] = f"Image quality issue: {str(e)}. Please upload a clearer image."
            
        except ImageValidationError as e:
            logger.warning(f"Image validation failed for {classifier_type.value}: {str(e)}")
            context['error'] = f"Image validation failed: {str(e)}"
            
        except ClassificationError as e:
            logger.error(f"Classification failed for {classifier_type.value}: {str(e)}")
            context['error'] = f"Failed to analyze image: {str(e)}"
            
        except Exception as e:
            logger.exception(f"Unexpected error in {classifier_type.value} classification")
            context['error'] = "An unexpected error occurred. Please try again with a different image."
            
        finally:
            # Clean up temporary file
            if image_path:
                try:
                    os.remove(image_path)
                except OSError:
                    pass
    
    return render(request, template_name, context)


def bones_detect_view(request):
    """Detect bone fractures from X-ray image."""
    return _process_ai_classification(
        request, 
        ClassifierType.BONES, 
        'ai_classifier/new_bones_classifier.html'
    )


def brain_detect_view(request):
    """Detect brain tumors from MRI image."""
    return _process_ai_classification(
        request, 
        ClassifierType.BRAIN, 
        'ai_classifier/new_brain_classifier.html'
    )


def ai_classify_api(request):
    """
    REST API endpoint for medical classification.
    
    Accepts POST requests with:
    - image: The image file to classify
    - classifier: One of 'skin', 'brain', 'bones'
    - generate_heatmap (optional): 'true' to generate Grad-CAM heatmap
    - use_tta (optional): 'true' to use test-time augmentation
    - assess_quality (optional): 'true' to assess image quality
    
    Returns JSON with classification results.
    """
    if request.method != 'POST':
        return JsonResponse({
            'success': False,
            'error': 'Only POST method is allowed',
            'supported_classifiers': ['skin', 'brain', 'bones']
        }, status=405)
    
    if not request.FILES.get('image'):
        return JsonResponse({
            'success': False,
            'error': 'No image file provided'
        }, status=400)
    
    # Get classifier type
    classifier_name = request.POST.get('classifier', 'brain').lower()
    classifier_map = {
        'brain': ClassifierType.BRAIN,
        'bones': ClassifierType.BONES,
        'xray': ClassifierType.BONES,  # Alias
    }
    
    if classifier_name not in classifier_map:
        return JsonResponse({
            'success': False,
            'error': f'Invalid classifier: {classifier_name}',
            'supported_classifiers': list(classifier_map.keys())
        }, status=400)
    
    classifier_type = classifier_map[classifier_name]
    image_path = None
    
    try:
        # Load and validate image
        image = request.FILES['image']
        img = Image.open(io.BytesIO(image.read()))
        
        if img.mode not in ('RGB', 'L', 'RGBA'):
            img = img.convert('RGB')
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            image_path = tmp_file.name
            img.save(image_path, format='PNG')
        
        # Get options from request
        generate_heatmap = request.POST.get('generate_heatmap', 'false').lower() == 'true'
        use_tta = request.POST.get('use_tta', 'false').lower() == 'true'
        assess_quality = request.POST.get('assess_quality', 'true').lower() == 'true'
        
        # Perform classification
        result = classify_image(
            image_path,
            classifier_type,
            generate_heatmap=generate_heatmap,
            top_n=5,
            use_tta=use_tta,
            assess_quality=assess_quality
        )
        
        # Build response
        response_data = {
            'success': True,
            'classifier': classifier_type.value,
            'result': {
                'class_name': result.class_name,
                'confidence': result.confidence,
                'is_confident': result.is_confident,
                'severity': result.severity.value,
                'urgency': result.urgency,
                'description': result.description,
            },
            'top_predictions': [
                {
                    'class_name': pred.class_name,
                    'probability': pred.probability,
                    'description': pred.description,
                    'severity': pred.severity.value if pred.severity else None,
                }
                for pred in result.top_predictions
            ],
            'recommendations': result.recommendations,
            'preprocessing_applied': result.preprocessing_applied,
            'inference_time_ms': result.inference_time_ms,
            'image_hash': result.image_hash,
        }
        
        # Add quality assessment if available
        if result.quality_assessment:
            qa = result.quality_assessment
            response_data['quality_assessment'] = {
                'overall_quality': qa.overall_quality.value,
                'quality_score': qa.quality_score,
                'brightness_score': qa.brightness_score,
                'contrast_score': qa.contrast_score,
                'sharpness_score': qa.sharpness_score,
                'noise_score': qa.noise_score,
                'is_acceptable': qa.is_acceptable,
                'issues': qa.issues,
            }
        
        # Add heatmap if generated
        if result.heatmap_base64:
            response_data['heatmap_base64'] = result.heatmap_base64
        
        return JsonResponse(response_data)
        
    except ModelNotFoundError as e:
        return JsonResponse({
            'success': False,
            'error': f'Classification model not available: {str(e)}'
        }, status=503)
        
    except (ImageValidationError, ImageQualityError) as e:
        return JsonResponse({
            'success': False,
            'error': f'Image validation failed: {str(e)}'
        }, status=400)
        
    except ClassificationError as e:
        return JsonResponse({
            'success': False,
            'error': f'Classification failed: {str(e)}'
        }, status=500)
        
    except Exception as e:
        logger.exception(f"Unexpected error in API classification")
        return JsonResponse({
            'success': False,
            'error': 'An unexpected error occurred'
        }, status=500)
        
    finally:
        if image_path:
            try:
                os.remove(image_path)
            except OSError:
                pass


def ai_models_status_api(request):
    """
    API endpoint to check classification models status.
    
    Returns JSON with status of all classification models.
    """
    try:
        status = check_models_status()
        model_info = {}
        
        for classifier_type in ClassifierType:
            info = get_model_info(classifier_type)
            conditions = get_all_conditions(classifier_type)
            model_info[classifier_type.value] = {
                'info': info,
                'conditions': conditions,
            }
        
        return JsonResponse({
            'success': True,
            'models_status': status,
            'model_details': model_info,
        })
        
    except Exception as e:
        logger.exception("Error checking models status")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


def ai_models_status_view(request):
    """API endpoint to check AI models status."""
    status = check_models_status()
    return JsonResponse(status)
