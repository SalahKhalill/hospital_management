GENDER_CHOICES = [("Male", "Male"), ("Female", "Female")] 

GOVERNORATE_CHOICES = [
    ('Adana', 'Adana'),
    ('Ankara', 'Ankara'),
    ('Antalya', 'Antalya'),
    ('Aydın', 'Aydın'),
    ('Balıkesir', 'Balıkesir'),
    ('Bursa', 'Bursa'),
    ('Denizli', 'Denizli'),
    ('Diyarbakır', 'Diyarbakır'),
    ('Edirne', 'Edirne'),
    ('Eskişehir', 'Eskişehir'),
    ('Gaziantep', 'Gaziantep'),
    ('Hatay', 'Hatay'),
    ('Istanbul', 'Istanbul'),
    ('Izmir', 'Izmir'),
    ('Kayseri', 'Kayseri'),
    ('Kocaeli', 'Kocaeli'),
    ('Konya', 'Konya'),
    ('Malatya', 'Malatya'),
    ('Manisa', 'Manisa'),
    ('Mersin', 'Mersin'),
    ('Muğla', 'Muğla'),
    ('Sakarya', 'Sakarya'),
    ('Sivas', 'Sivas'),
    ('Samsun', 'Samsun'),
    ('Şanlıurfa', 'Şanlıurfa'),
    ('Tekirdağ', 'Tekirdağ'),
    ('Trabzon', 'Trabzon'),
    ('Van', 'Van'),
]


PERIOD_CHOICES = (
        ('10-11', '10 AM - 11 AM'),
        ('11-12', '11 AM - 12 PM'),
        ('12-1', '12 PM - 1 PM'),
        
        ('4-5', '4 PM - 5 PM'),
        ('5-6', '5 PM - 6 PM'),
        ('6-7', '6 PM - 7 PM'),
    )

SHIFT_PERIOD_CHOICES = (
        ('Morning Shift', 'Morning Shift'),
        ('Evening Shift', 'Evening Shift'),
    )

DOCTOR_SPECIALIZE_CHOICES = (
        ('Consultant', 'Consultant'),
        ('Specialist', 'Specialist'),
        ('General Practitioner', 'General Practitioner'),
    )