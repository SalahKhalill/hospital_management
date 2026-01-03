# Hospital Management System

A comprehensive hospital management system built with Django, featuring deep learning-powered medical image classification for brain tumors and bone fractures.

## Features

- **User Management**: Multi-role system (Admin, Doctor, Nurse, Patient)
- **Appointment System**: Schedule and manage patient appointments
- **Department Management**: Organize hospital departments and staff
- **Medicine Inventory**: Track and manage medicine stock
- **Medical Diagnostics**: Image classification for:
   - Brain tumor detection (MRI scans)
   - Bone fracture detection (X-rays)

## Quick Start

### Prerequisites

- Python 3.10+
- pip (Python package manager)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SalahKhalill/hospital_management.git
   cd hospital_management
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env with your settings (generate a new SECRET_KEY!)
   ```

5. **Download AI Models**
   
   Download the model files from Google Drive (links in `AI Models drive links.txt`) and place them in the `models/` folder:
   - `brain.h5` - Brain tumor classifier
   - `skin.h5` - Skin condition classifier
   - `xray.h5` - Bone fracture classifier

6. **Run migrations**
   ```bash
   python manage.py migrate
   ```

7. **Create a superuser**
   ```bash
   python manage.py createsuperuser
   ```

8. **Run the development server**
   ```bash
   python manage.py runserver
   ```

9. **Access the application**
   - Main site: http://127.0.0.1:8000
   - Admin panel: http://127.0.0.1:8000/admin

## Project Structure

```
hospital_management/
├── Hospital/               # Django project settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── master/                 # Main application
│   ├── ai_classifier.py    # Unified AI classification module
│   ├── models.py           # Database models
│   ├── views.py            # View functions
│   ├── forms.py            # Django forms
│   ├── templates/          # HTML templates
│   └── static/             # CSS, JS, images
├── models/                 # AI model files (.h5)
├── media/                  # User uploaded files
├── logs/                   # Application logs
├── requirements.txt
├── .env.example
└── manage.py
```

## Security Features

- Secure session cookies (HTTPS in production)
- CSRF protection on all forms
- HTTP Strict Transport Security (HSTS)
- XSS and clickjacking protection
- Argon2 password hashing
- Environment-based configuration

## AI Models

The system uses TensorFlow/Keras models for medical image classification:

| Model | Purpose | Classes |
|-------|---------|---------|
| brain.h5 | Brain MRI analysis | No Tumor, Stable Tumor, Unstable Tumor |
| skin.h5 | Skin condition detection | 9 skin conditions including melanoma |
| xray.h5 | Bone X-ray analysis | No Fracture, Fractured |

**Note**: AI models are loaded lazily to improve startup time.

### Advanced AI Features

- **Test-Time Augmentation (TTA)**: Multiple augmented predictions for more robust results
- **Image Quality Assessment**: Automatic evaluation of brightness, contrast, sharpness, and noise
- **Grad-CAM Visualization**: Heatmaps showing which regions influenced the AI decision
- **Severity Classification**: Automatic urgency and severity assessment
- **Comprehensive Recommendations**: Medical guidance based on detected conditions

## REST API

### Classification Endpoint

`POST /api/ai/classify/`

Classify medical images using AI models.

**Parameters:**
- `image` (file, required): The image file to classify
- `classifier` (string, required): One of `skin`, `brain`, `bones`
- `generate_heatmap` (string, optional): `true` to generate Grad-CAM heatmap
- `use_tta` (string, optional): `true` to use test-time augmentation
- `assess_quality` (string, optional): `true` to assess image quality (default: true)

**Example Response:**
```json
{
  "success": true,
  "classifier": "skin",
  "result": {
    "class_name": "Melanoma",
    "confidence": 0.92,
    "is_confident": true,
    "severity": "critical",
    "urgency": "Immediate attention required",
    "description": "Malignant skin cancer..."
  },
  "top_predictions": [...],
  "recommendations": [...],
  "quality_assessment": {...},
  "inference_time_ms": 245
}
```

### Models Status Endpoint

`GET /api/ai/models/status/`

Check the status of all AI models.

**Example Response:**
```json
{
  "success": true,
  "models_status": {
    "skin": {"available": true, "path": "..."},
    "brain": {"available": true, "path": "..."},
    "bones": {"available": true, "path": "..."}
  }
}
```

## User Roles

| Role | Capabilities |
|------|-------------|
| **Admin** | Full system access, user management, reports |
| **Doctor** | Patient management, appointments, AI diagnostics |
| **Nurse** | Patient care, appointment scheduling |
| **Patient** | View appointments, medical records |

## Development

### Running Tests
```bash
python manage.py test
```

### Code Style
Follow PEP 8 guidelines for Python code.

## Email Configuration

For password reset and notifications, configure SMTP in `.env`:
```
EMAIL_HOST_USER=your-email@gmail.com
EMAIL_HOST_PASSWORD=your-app-password
```

For Gmail, use an [App Password](https://support.google.com/accounts/answer/185833).

## Contributing

1. Create a new branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Commit: `git commit -m "Add your feature"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

## License

This project is for educational purposes.

## Disclaimer

The AI diagnostic tools are for educational/demonstration purposes only and should **NOT** be used for actual medical diagnosis. Always consult qualified healthcare professionals.
