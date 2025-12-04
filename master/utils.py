
# instance refere to doctor Model
def doctors_directory_path(instance, filename):
    return f"doctors/doctor_{instance.id}/{filename}"


# instance refere to nurse Model
def nurses_directory_path(instance, filename):
    return f"nurses/nurse_{instance.id}/{filename}"


# instance refere to patient Model
def patients_directory_path(instance, filename):
    return f"patients/patient_{instance.id}/{filename}"


# instance refere to department Model
def department_directory_path(instance, filename):
    return f"departments/department_{instance.id}/{filename}"