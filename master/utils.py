
def doctors_directory_path(instance, filename):
    return f"doctors/doctor_{instance.id}/{filename}"


def nurses_directory_path(instance, filename):
    return f"nurses/nurse_{instance.id}/{filename}"


def patients_directory_path(instance, filename):
    return f"patients/patient_{instance.id}/{filename}"


def department_directory_path(instance, filename):
    return f"departments/department_{instance.id}/{filename}"