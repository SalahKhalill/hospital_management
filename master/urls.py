from django.urls import path, include
from . import views
from django.contrib.auth import views as auth_views 


urlpatterns = [
    path('', views.home_view, name='home'),
    
    path("<str:hospital_user>/login/", views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('after_login/', views.after_login_view, name='after_login'),
    path('<str:hospital_user>/register/', views.register_view, name='register'),
    
    path('admin_profile/', views.AdminProfileView.as_view(), name='admin_profile'),
    path('user_profile/', views.UserProfileView.as_view(), name='user_profile'),
    
    path('change_password/', views.change_password_view, name='change_password'),
    
    path('password_reset/',auth_views.PasswordResetView.as_view(template_name= "password_reset_form.html"),name='password_reset'),
    path('password_reset/done/',auth_views.PasswordResetDoneView.as_view(template_name = 'password_reset_sent.html'),name='password_reset_done'),
    path('reset/<uidb64>/<token>/',auth_views.PasswordResetConfirmView.as_view(template_name = 'password_reset_confirm.html'),name='password_reset_confirm'),
    path('reset/done/',auth_views.PasswordResetCompleteView.as_view(template_name = 'password_reset_complete.html'),name='password_reset_complete'),
    
    path('admin_dashboard/', views.admin_dashboard_view, name='admin_dashboard'),
    path('doctor_dashboard/', views.doctor_dashboard_view, name='doctor_dashboard'),
    path('nurse_dashboard/', views.nurse_dashboard_view, name='nurse_dashboard'),
    path('patient_dashboard/', views.patient_dashboard_view, name='patient_dashboard'),
    
    path('admin_manager/<str:data>/', views.admin_manager_view, name='admin_manager'),
    
    path('admin_add_user/<str:hospital_user>/', views.admin_add_user_view, name='admin_add_user'),
    path('admin_confirm_user/<int:id>/', views.admin_confirm_user_view, name='admin_confirm_user'),
    path('admin_delete_user/<int:id>/', views.admin_delete_user_view, name='admin_delete_user'),
    path('admin_update_user/<int:id>/', views.AdminUpdateUserView.as_view(), name='admin_update_user'),
    
    path('admin_add_appointment/', views.admin_add_appointment_view, name='admin_add_appointment'),
    path('admin_confirm_appointment/<int:id>/', views.admin_confirm_appointment_view, name='admin_confirm_appointment'),
    path('admin_delete_appointment/<int:id>/', views.admin_delete_appointment_view, name='admin_delete_appointment'),
    
    
    path('admin_add_department/', views.admin_add_department_view, name='admin_add_department'),
    path('admin_update_department/<int:id>/', views.admin_update_department_view, name='admin_update_department'),
    path('admin_delete_department/<int:id>/', views.admin_delete_department_view, name='admin_delete_department'),
    
    path('admin_add_medicine/', views.admin_add_medicine_view, name='admin_add_medicine'),
    path('admin_update_medicine/<int:id>/', views.admin_update_medicine_view, name='admin_update_medicine'),
    path('admin_delete_medicine/<int:id>/', views.admin_delete_medicine_view, name='admin_delete_medicine'),
    
    
    path('doctor_appointment/', views.doctor_appointment_view, name='doctor_appointment'),
    path('nurse_appointment/', views.nurse_appointment_view, name='nurse_appointment'),
    path('patient_appointment/', views.patient_appointment_view, name='patient_appointment'),
    
    path('patient_update_appointment/<int:id>/', views.patient_update_appointment_view, name='patient_update_appointment'),
    path('patient_delete_appointment/<int:id>/', views.patient_delete_appointment_view, name='patient_delete_appointment'),
    
    
    path('book_appointment/', views.book_appointment_view, name='book_appointment'),
    
    path('doctor_report/<int:id>/', views.doctor_report_view, name='doctor_report'),
    path('admin_discharge_patient/<int:id>/', views.admin_discharge_patient_view, name='admin_discharge_patient'),
    
    path('final_discharge/<int:id>/', views.final_discharge_view, name='final_discharge'),
    path('pay_discharge/<int:id>/', views.pay_discharge_view, name='pay_discharge'),
    path('download_permit_pdf/<int:id>/', views.download_permit_pdf, name='download_permit_pdf'),
    
    path('department/<int:id>/', views.department_view, name='department'),
    path('medicine/', views.medicine_view, name='medicine'),
    
    path('aboutus/', views.aboutus_view, name='aboutus'),
    path('contactus/', views.contactus_view, name='contactus'),
    path('send_mail/<int:id>/', views.send_mail_view, name='send_mail'),


    #---------FOR AI RELATED URLS-------------------------------------
    path('skin_detect/', views.skin_detect_view, name='skin_detect'),
    path('bones_classifier/', views.bones_detect_view, name='bones_classifier'),
    path('brain_classifier/', views.brain_detect_view, name='brain_classifier'),

]

