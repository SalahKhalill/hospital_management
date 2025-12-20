"""
Advanced Medical Classifier Module for Medical Image Analysis
=========================================================

Enterprise-grade medical image classification with comprehensive features:

Core Features:
- Lazy model loading for fast startup
- Advanced image preprocessing pipeline
- Test-Time Augmentation (TTA) for robust predictions
- Confidence thresholds with uncertainty quantification
- Grad-CAM heatmap visualization for explainability

Advanced Features:
- Image quality assessment
- Automatic region of interest detection
- Multi-scale analysis
- Prediction caching for performance
- Detailed medical context and recommendations
- Inference time tracking
- Comprehensive result serialization
"""
import numpy as np
import cv2
import os
import time
import hashlib
import json
from django.conf import settings
from typing import Tuple, List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
import logging
import base64
from datetime import datetime
from tensorflow.keras.applications.vgg16 import preprocess_input

logger = logging.getLogger(__name__)

# Import brain_classifier for specialized preprocessing
try:
    from master import brain_classifier
    BRAIN_CLASSIFIER_AVAILABLE = True
except ImportError:
    BRAIN_CLASSIFIER_AVAILABLE = False
    logger.warning("brain_classifier module not available")


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class ClassifierType(Enum):
    BRAIN = "brain"
    BONES = "xray"


class Severity(Enum):
    """Medical severity levels for classification results."""
    NORMAL = "normal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    
    @property
    def priority(self) -> int:
        """Get numeric priority for sorting."""
        return {
            Severity.NORMAL: 0,
            Severity.LOW: 1,
            Severity.MODERATE: 2,
            Severity.HIGH: 3,
            Severity.CRITICAL: 4,
        }[self]


class ImageQuality(Enum):
    """Image quality assessment results."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNUSABLE = "unusable"


@dataclass
class QualityAssessment:
    """Detailed image quality analysis."""
    overall_quality: ImageQuality
    brightness_score: float  # 0-1, 0.5 is ideal
    contrast_score: float    # 0-1, higher is better
    sharpness_score: float   # 0-1, higher is better
    noise_score: float       # 0-1, lower noise is better
    issues: List[str]
    suggestions: List[str]
    
    @property
    def is_usable(self) -> bool:
        return self.overall_quality != ImageQuality.UNUSABLE


@dataclass
class PredictionDetail:
    """Individual prediction with class and probability."""
    class_name: str
    probability: float
    severity: Severity
    description: str = ""
    
    def __str__(self):
        return f"{self.class_name}: {self.probability * 100:.1f}%"


@dataclass
class ClassificationResult:
    """Comprehensive result from image classification."""
    # Core results
    class_index: int
    class_name: str
    confidence: float
    severity: Severity
    is_confident: bool
    
    # Detailed predictions
    top_predictions: List[PredictionDetail]
    all_probabilities: List[float]
    
    # Medical context
    recommendations: List[str]
    medical_info: Dict[str, str]
    urgency_level: str
    
    # Technical details
    preprocessing_applied: List[str] = field(default_factory=list)
    inference_time_ms: float = 0.0
    model_version: str = "1.0"
    tta_enabled: bool = False
    
    # Visualizations
    heatmap_base64: Optional[str] = None
    roi_base64: Optional[str] = None
    
    # Quality assessment
    quality_assessment: Optional[QualityAssessment] = None
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    image_hash: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            'class_index': self.class_index,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'confidence_percent': f"{self.confidence * 100:.1f}%",
            'severity': self.severity.value,
            'severity_priority': self.severity.priority,
            'is_confident': self.is_confident,
            'top_predictions': [
                {
                    'class': p.class_name,
                    'probability': p.probability,
                    'probability_percent': f"{p.probability * 100:.1f}%",
                    'severity': p.severity.value,
                    'description': p.description,
                }
                for p in self.top_predictions
            ],
            'recommendations': self.recommendations,
            'medical_info': self.medical_info,
            'urgency_level': self.urgency_level,
            'preprocessing_applied': self.preprocessing_applied,
            'inference_time_ms': round(self.inference_time_ms, 2),
            'model_version': self.model_version,
            'tta_enabled': self.tta_enabled,
            'has_heatmap': self.heatmap_base64 is not None,
            'has_roi': self.roi_base64 is not None,
            'quality': self.quality_assessment.overall_quality.value if self.quality_assessment else None,
            'timestamp': self.timestamp,
        }
    
    def get_summary(self) -> str:
        """Get a human-readable summary."""
        return (
            f"Diagnosis: {self.class_name} ({self.confidence*100:.1f}% confidence)\n"
            f"Severity: {self.severity.value.upper()}\n"
            f"Urgency: {self.urgency_level}\n"
            f"Analysis time: {self.inference_time_ms:.0f}ms"
        )


# ============================================================================
# EXCEPTIONS
# ============================================================================

class ModelNotFoundError(Exception):
    """Raised when a model file cannot be found."""
    pass


class ClassificationError(Exception):
    """Raised when classification fails."""
    pass


class ImageValidationError(Exception):
    """Raised when image validation fails."""
    pass


class ImageQualityError(Exception):
    """Raised when image quality is too poor for analysis."""
    pass


# ============================================================================
# CLASSIFIER CONFIGURATION
# ============================================================================

CLASSIFIER_CONFIG = {
<<<<<<< HEAD
=======
    ClassifierType.SKIN: {
        'classes': [
            'Actinic Keratosis',
            'Basal Cell Carcinoma',
            'Dermatofibroma',
            'Melanoma',
            'Nevus',
            'Pigmented Benign Keratosis',
            'Seborrheic Keratosis',
            'Squamous Cell Carcinoma',
            'Vascular Lesion'
        ],
        'severity_mapping': {
            0: Severity.MODERATE,
            1: Severity.HIGH,
            2: Severity.LOW,
            3: Severity.CRITICAL,
            4: Severity.NORMAL,
            5: Severity.LOW,
            6: Severity.LOW,
            7: Severity.HIGH,
            8: Severity.LOW,
        },
        'urgency_mapping': {
            0: "Schedule appointment within 2 weeks",
            1: "Schedule appointment within 1 week",
            2: "Routine follow-up",
            3: "URGENT - Seek immediate consultation",
            4: "Annual monitoring",
            5: "Routine follow-up",
            6: "Routine follow-up",
            7: "Schedule appointment within 1 week",
            8: "Routine follow-up",
        },
        'descriptions': {
            0: "Precancerous scaly patch caused by sun damage. Can develop into squamous cell carcinoma if untreated.",
            1: "Most common form of skin cancer. Usually grows slowly and rarely spreads but requires treatment.",
            2: "Benign fibrous skin nodule. Usually harmless and doesn't require treatment.",
            3: "Most dangerous form of skin cancer. Develops from pigment-producing cells. Early detection is crucial.",
            4: "Common benign mole. Monitor for changes using the ABCDE rule.",
            5: "Benign pigmented skin growth. No treatment necessary unless cosmetically desired.",
            6: "Common benign growth that appears with aging. Waxy, scaly appearance.",
            7: "Second most common skin cancer. Can spread if not treated early.",
            8: "Abnormality of blood vessels. Usually benign but may require evaluation.",
        },
        'recommendations': {
            0: [
                "Schedule dermatologist appointment within 2 weeks",
                "Avoid sun exposure on affected area",
                "Use broad-spectrum SPF 50+ sunscreen",
                "Document any changes with photos",
                "May require cryotherapy or topical treatment"
            ],
            1: [
                "Consult dermatologist within 1 week",
                "Surgical removal is typically recommended",
                "Excellent prognosis with early treatment",
                "Regular skin checks after treatment",
                "Protect skin from further sun damage"
            ],
            2: [
                "Usually no treatment necessary",
                "Monitor for any changes in size or appearance",
                "Can be removed if symptomatic or cosmetically desired",
                "Routine annual skin check recommended"
            ],
            3: [
                "⚠️ URGENT: Contact dermatologist IMMEDIATELY",
                "Do not delay - early treatment saves lives",
                "Prepare for possible biopsy",
                "Avoid sun exposure completely",
                "Document lesion with photos for comparison",
                "Consider genetic counseling if family history exists"
            ],
            4: [
                "Normal mole - no immediate concern",
                "Monitor using ABCDE rule monthly",
                "Take photos to track any changes",
                "Annual dermatologist skin check recommended",
                "Use sun protection to prevent changes"
            ],
            5: [
                "Benign lesion - no treatment needed",
                "Can be removed for cosmetic reasons",
                "Monitor for any changes",
                "Annual skin check recommended"
            ],
            6: [
                "Common benign growth with aging",
                "No treatment necessary unless irritated",
                "Can be removed if symptomatic",
                "Not associated with skin cancer"
            ],
            7: [
                "Schedule dermatologist appointment within 1 week",
                "Surgical removal typically recommended",
                "Good prognosis with early treatment",
                "Regular follow-up examinations needed",
                "Strict sun protection essential"
            ],
            8: [
                "Usually benign vascular anomaly",
                "Consult if bleeding or rapidly changing",
                "May be treated with laser therapy if desired",
                "Monitor for any changes"
            ],
        },
        'confidence_threshold': 0.60,
        'min_image_size': (50, 50),
        'preprocessing': ['denoise', 'normalize_lighting'],
        'tta_rotations': [0, 90, 180, 270],
        'tta_flips': [False, True],
    },
>>>>>>> origin/main
    ClassifierType.BRAIN: {
        'classes': [
            'No Tumor',
            'Tumor Detected',
        ],
        'is_binary': True,
        'severity_mapping': {
            0: Severity.NORMAL,    # no tumor
            1: Severity.CRITICAL,  # tumor detected
        },
        'urgency_mapping': {
            0: "Continue regular health monitoring",
            1: "URGENT - Tumor detected, immediate neurological consultation required",
        },
        'descriptions': {
            0: "No evidence of tumor detected in the MRI scan. Brain structure appears normal.",
            1: "Brain tumor detected. Immediate medical consultation is required for proper diagnosis and treatment planning.",
        },
        'recommendations': {
            0: [
                "No tumor detected - continue regular health checkups",
                "Report any new neurological symptoms promptly",
                "Maintain healthy lifestyle for brain health",
                "Consider follow-up MRI in 1-2 years if symptoms persist"
            ],
            1: [
                "⚠️ CRITICAL: Brain tumor detected - Immediate neurosurgical consultation required",
                "Contact neurologist or neuro-oncologist today for evaluation",
                "Prepare for additional advanced imaging (MRI with contrast, CT scan)",
                "Further testing needed to determine tumor type and grade",
                "Discuss treatment options with neurosurgeon",
                "Do not delay - timely intervention is crucial"
                "Monitor for symptoms: vision changes, hormonal changes"
            ],
        },
        'confidence_threshold': 0.60,
        'min_image_size': (64, 64),
        'preprocessing': ['denoise', 'enhance_contrast'],
        'tta_rotations': [0],
        'tta_flips': [False, True],
    },
    ClassifierType.BONES: {
        'classes': [
            'No Fracture',
            'Fractured'
        ],
<<<<<<< HEAD
        'is_binary': True,
=======
>>>>>>> origin/main
        'severity_mapping': {
            0: Severity.NORMAL,
            1: Severity.HIGH,
        },
        'urgency_mapping': {
            0: "Follow up if pain persists",
            1: "Seek orthopedic consultation today",
        },
        'descriptions': {
            0: "No fracture detected in the X-ray. Bone structure appears intact.",
            1: "Fracture detected. The bone shows a break or crack that requires medical attention.",
        },
        'recommendations': {
            0: [
                "No fracture detected in this image",
                "If pain persists, consider repeat imaging in 7-10 days",
                "Stress fractures may not show on initial X-ray",
                "Consider MRI if high clinical suspicion remains",
                "Apply RICE (Rest, Ice, Compression, Elevation) for soft tissue injuries"
            ],
            1: [
                "Fracture detected - seek medical attention",
                "Immobilize the affected area immediately",
                "Avoid putting weight on the injured limb",
                "Consult orthopedic specialist today",
                "May require casting, splinting, or surgical intervention",
                "Follow-up X-ray needed to monitor healing"
            ],
        },
        'confidence_threshold': 0.65,
        'min_image_size': (64, 64),
        'preprocessing': ['denoise', 'enhance_edges', 'enhance_contrast'],
        'tta_rotations': [0],
        'tta_flips': [False, True],
    },
}


# ============================================================================
# IMAGE QUALITY ASSESSMENT
# ============================================================================

class ImageQualityAnalyzer:
    """Comprehensive image quality analysis for medical images."""
    
    @staticmethod
    def calculate_brightness(image: np.ndarray) -> float:
        """Calculate normalized brightness score (0.5 is ideal)."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        mean_brightness = np.mean(gray) / 255.0
        # Convert to score where 0.5 (middle brightness) is best
        return 1.0 - abs(mean_brightness - 0.5) * 2
    
    @staticmethod
    def calculate_contrast(image: np.ndarray) -> float:
        """Calculate contrast score using standard deviation."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        std = np.std(gray)
        # Normalize to 0-1 range (std of 50+ is good contrast)
        return min(std / 50.0, 1.0)
    
    @staticmethod
    def calculate_sharpness(image: np.ndarray) -> float:
        """Calculate sharpness using Laplacian variance."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalize to 0-1 range (variance of 100+ is sharp)
        return min(laplacian_var / 100.0, 1.0)
    
    @staticmethod
    def calculate_noise(image: np.ndarray) -> float:
        """Estimate noise level in image."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Use high-pass filter to estimate noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = np.abs(gray.astype(float) - blur.astype(float))
        noise_level = np.mean(noise)
        
        # Lower noise is better, normalize inversely
        return max(0, 1.0 - noise_level / 20.0)
    
    @classmethod
    def assess_quality(cls, image: np.ndarray, classifier_type: ClassifierType) -> QualityAssessment:
        """Perform comprehensive quality assessment."""
        brightness = cls.calculate_brightness(image)
        contrast = cls.calculate_contrast(image)
        sharpness = cls.calculate_sharpness(image)
        noise = cls.calculate_noise(image)
        
        issues = []
        suggestions = []
        
        # Analyze each metric
        if brightness < 0.3:
            issues.append("Image is too dark or too bright")
            suggestions.append("Ensure proper lighting when capturing the image")
        
        if contrast < 0.4:
            issues.append("Low contrast detected")
            suggestions.append("Improve image contrast or lighting conditions")
        
        if sharpness < 0.3:
            issues.append("Image appears blurry")
            suggestions.append("Hold camera steady and ensure proper focus")
        
        if noise < 0.5:
            issues.append("High noise level detected")
            suggestions.append("Use better lighting to reduce noise")
        
        # Calculate overall score
        overall_score = (brightness * 0.2 + contrast * 0.3 + sharpness * 0.35 + noise * 0.15)
        
        # Determine quality level
        if overall_score >= 0.8:
            quality = ImageQuality.EXCELLENT
        elif overall_score >= 0.6:
            quality = ImageQuality.GOOD
        elif overall_score >= 0.4:
            quality = ImageQuality.FAIR
        elif overall_score >= 0.2:
            quality = ImageQuality.POOR
            issues.append("Image quality is poor - results may be unreliable")
        else:
            quality = ImageQuality.UNUSABLE
            issues.append("Image quality too poor for reliable analysis")
            suggestions.append("Please retake the image with better conditions")
        
        return QualityAssessment(
            overall_quality=quality,
            brightness_score=brightness,
            contrast_score=contrast,
            sharpness_score=sharpness,
            noise_score=noise,
            issues=issues,
            suggestions=suggestions,
        )


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

class ImagePreprocessor:
    """Advanced image preprocessing for medical images."""
    
    @staticmethod
    def validate_image(image: np.ndarray, classifier_type: ClassifierType) -> None:
        """Validate image meets minimum requirements."""
        if image is None:
            raise ImageValidationError("Failed to load image")
        
        config = CLASSIFIER_CONFIG[classifier_type]
        min_size = config.get('min_image_size', (50, 50))
        
        if image.shape[0] < min_size[0] or image.shape[1] < min_size[1]:
            raise ImageValidationError(
                f"Image too small. Minimum size: {min_size[0]}x{min_size[1]} pixels. "
                f"Your image: {image.shape[1]}x{image.shape[0]} pixels"
            )
        
        if len(image.shape) < 2:
            raise ImageValidationError("Invalid image format - must be 2D or 3D array")
    
    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE."""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    @staticmethod
    def enhance_edges(image: np.ndarray) -> np.ndarray:
        """Enhance edges for better fracture detection."""
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    
    @staticmethod
    def normalize_lighting(image: np.ndarray) -> np.ndarray:
        """Normalize lighting conditions."""
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v = cv2.equalizeHist(v)
            hsv = cv2.merge([h, s, v])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return cv2.equalizeHist(image)
    
    @staticmethod
    def denoise(image: np.ndarray) -> np.ndarray:
        """Remove noise from image."""
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 21)
        return cv2.fastNlMeansDenoising(image, None, 6, 7, 21)
    
    @staticmethod
    def auto_crop_roi(image: np.ndarray, classifier_type: ClassifierType) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Automatically detect and crop to region of interest."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold to find main content
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image, (0, 0, image.shape[1], image.shape[0])
        
        # Get bounding box of largest contour
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        
        # Add padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        cropped = image[y:y+h, x:x+w]
        return cropped, (x, y, w, h)
    
    @classmethod
    def preprocess(cls, image: np.ndarray, classifier_type: ClassifierType) -> Tuple[np.ndarray, List[str]]:
        """Apply full preprocessing pipeline based on classifier type."""
        cls.validate_image(image, classifier_type)
        
        config = CLASSIFIER_CONFIG[classifier_type]
        preprocessing_steps = config.get('preprocessing', [])
        applied = []
        processed = image.copy()
        
        for step in preprocessing_steps:
            if step == 'denoise':
                processed = cls.denoise(processed)
                applied.append("noise_reduction")
            elif step == 'enhance_contrast':
                processed = cls.enhance_contrast(processed)
                applied.append("contrast_enhancement")
            elif step == 'enhance_edges':
                processed = cls.enhance_edges(processed)
                applied.append("edge_enhancement")
            elif step == 'normalize_lighting':
                processed = cls.normalize_lighting(processed)
                applied.append("lighting_normalization")
        
        return processed, applied


# ============================================================================
# TEST-TIME AUGMENTATION
# ============================================================================

class TestTimeAugmentation:
    """Apply test-time augmentation for more robust predictions."""
    
    @staticmethod
    def augment_image(image: np.ndarray, rotation: int = 0, flip: bool = False) -> np.ndarray:
        """Apply augmentation to image."""
        augmented = image.copy()
        
        if rotation != 0:
            if rotation == 90:
                augmented = cv2.rotate(augmented, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                augmented = cv2.rotate(augmented, cv2.ROTATE_180)
            elif rotation == 270:
                augmented = cv2.rotate(augmented, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        if flip:
            augmented = cv2.flip(augmented, 1)  # Horizontal flip
        
        return augmented
    
    @classmethod
    def get_augmented_predictions(
        cls,
        model,
        image: np.ndarray,
        input_shape: Tuple[int, int],
        classifier_type: ClassifierType
    ) -> np.ndarray:
        """Get averaged predictions from multiple augmented versions."""
        config = CLASSIFIER_CONFIG[classifier_type]
        rotations = config.get('tta_rotations', [0])
        flips = config.get('tta_flips', [False])
        
        all_predictions = []
        
        for rotation in rotations:
            for flip in flips:
                # Augment image
                augmented = cls.augment_image(image, rotation, flip)
                
                # Resize and prepare
                resized = cv2.resize(augmented, dsize=input_shape)
                img_expanded = np.expand_dims(resized, axis=0).astype(np.float32)
                img_normalized = img_expanded / 255.0
                
                # Predict
                pred = model.predict(img_normalized, verbose=0)
                all_predictions.append(pred[0])
        
        # Average predictions
        return np.mean(all_predictions, axis=0)


# ============================================================================
# GRAD-CAM VISUALIZATION
# ============================================================================

class GradCAMGenerator:
    """Generate Grad-CAM heatmaps for model interpretability."""
    
    @staticmethod
    def find_last_conv_layer(model) -> Optional[str]:
        """Find the last convolutional layer in the model."""
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                return layer.name
        return None
    
    @staticmethod
    def generate_heatmap(model, image: np.ndarray, class_idx: int) -> Optional[np.ndarray]:
        """Generate Grad-CAM heatmap."""
        try:
            import tensorflow as tf
            
            last_conv_layer_name = GradCAMGenerator.find_last_conv_layer(model)
            if last_conv_layer_name is None:
                return None
            
            last_conv_layer = model.get_layer(last_conv_layer_name)
            
            grad_model = tf.keras.models.Model(
                [model.inputs],
                [last_conv_layer.output, model.output]
            )
            
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(image)
                loss = predictions[:, class_idx]
            
            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            
            heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
            return heatmap.numpy()
            
        except Exception as e:
            logger.warning(f"Failed to generate Grad-CAM: {str(e)}")
            return None
    
    @staticmethod
    def overlay_heatmap(
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """Overlay heatmap on original image."""
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap, colormap)
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        return cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    @staticmethod
    def create_comparison_image(original: np.ndarray, heatmap_overlay: np.ndarray) -> np.ndarray:
        """Create side-by-side comparison of original and heatmap."""
        if len(original.shape) == 2:
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        
        # Resize to same height
        height = max(original.shape[0], heatmap_overlay.shape[0])
        
        original_resized = cv2.resize(original, (int(original.shape[1] * height / original.shape[0]), height))
        heatmap_resized = cv2.resize(heatmap_overlay, (int(heatmap_overlay.shape[1] * height / heatmap_overlay.shape[0]), height))
        
        # Add labels
        cv2.putText(original_resized, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
<<<<<<< HEAD
        cv2.putText(heatmap_resized, "Focus", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
=======
        cv2.putText(heatmap_resized, "AI Focus", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
>>>>>>> origin/main
        
        return np.hstack([original_resized, heatmap_resized])


# ============================================================================
# LAZY MODEL LOADER
# ============================================================================

class LazyModelLoader:
    """Lazy loader for TensorFlow models with caching."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models = {}
            cls._instance._input_shapes = {}
            cls._instance._tf = None
            cls._instance._load_times = {}
        return cls._instance
    
    def _get_tensorflow(self):
        """Lazy import TensorFlow."""
        if self._tf is None:
            import tensorflow as tf
            self._tf = tf
            # Suppress TensorFlow warnings
            tf.get_logger().setLevel('ERROR')
        return self._tf
    
    def _get_model_path(self, classifier_type: ClassifierType) -> str:
        """Get the full path to a model file."""
        model_filename = f"{classifier_type.value}.h5"
        return os.path.join(settings.BASE_DIR, 'models', model_filename)
    
    def _load_model(self, classifier_type: ClassifierType):
        """Load a model if not already loaded."""
        if classifier_type not in self._models:
            model_path = self._get_model_path(classifier_type)
            
            if not os.path.exists(model_path):
                raise ModelNotFoundError(
                    f"Model file not found: {model_path}. "
                    f"Please download the {classifier_type.value}.h5 model from Google Drive "
                    f"and place it in the 'models' folder."
                )
            
            try:
                start_time = time.time()
                tf = self._get_tensorflow()
                
                # Try multiple loading strategies for compatibility with different Keras versions
                model = None
                load_errors = []
                
                # Strategy 1: Standard load with compile=False (most compatible)
                try:
                    model = tf.keras.models.load_model(model_path, compile=False)
                except Exception as e1:
                    load_errors.append(f"Standard load: {str(e1)}")
                    
                    # Strategy 2: Load with safe_mode=False for older models
                    try:
                        model = tf.keras.models.load_model(
                            model_path, 
                            compile=False,
                            safe_mode=False
                        )
                    except Exception as e2:
                        load_errors.append(f"Safe mode off: {str(e2)}")
                        
                        # Strategy 3: Use legacy H5 loading via weights
                        try:
                            import h5py
                            with h5py.File(model_path, 'r') as f:
                                # Check if it's a weights-only file or full model
                                if 'model_config' in f.attrs:
                                    model_config = f.attrs.get('model_config')
                                    if isinstance(model_config, bytes):
                                        model_config = model_config.decode('utf-8')
                                    import json
                                    config = json.loads(model_config)
                                    
                                    # Reconstruct model from config
                                    model = tf.keras.models.model_from_config(config)
                                    model.load_weights(model_path)
                        except Exception as e3:
                            load_errors.append(f"H5 reconstruction: {str(e3)}")
                            
                            # Strategy 4: Custom objects for legacy layers
                            try:
                                custom_objects = {
                                    'relu6': tf.keras.activations.relu,
                                    'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D,
                                }
                                model = tf.keras.models.load_model(
                                    model_path,
                                    compile=False,
                                    custom_objects=custom_objects
                                )
                            except Exception as e4:
                                load_errors.append(f"Custom objects: {str(e4)}")
                
                if model is None:
                    error_details = "; ".join(load_errors)
                    raise ModelNotFoundError(
                        f"Failed to load {classifier_type.value} model after trying multiple strategies. "
                        f"The model may be incompatible with TensorFlow {tf.__version__}. "
                        f"Errors: {error_details}"
                    )
                
                # Get input shape safely
                try:
                    if hasattr(model, 'input_shape'):
                        input_shape = list(model.input_shape)
                    else:
                        input_shape = list(model.layers[0].input_shape)
                except:
                    # Default input shapes based on model type
                    default_shapes = {
<<<<<<< HEAD
=======
                        ClassifierType.SKIN: [None, 224, 224, 3],
>>>>>>> origin/main
                        ClassifierType.BRAIN: [None, 224, 224, 3],
                        ClassifierType.BONES: [None, 224, 224, 3],
                    }
                    input_shape = default_shapes.get(classifier_type, [None, 224, 224, 3])
                    logger.warning(f"Could not determine input shape for {classifier_type.value}, using default: {input_shape}")
                
                self._models[classifier_type] = model
                self._input_shapes[classifier_type] = tuple(input_shape[1:-1]) if len(input_shape) == 4 else (224, 224)
                self._load_times[classifier_type] = time.time() - start_time
                
                logger.info(
                    f"Loaded {classifier_type.value} model in {self._load_times[classifier_type]:.2f}s "
                    f"with input shape {input_shape}"
                )
            except ModelNotFoundError:
                raise
            except Exception as e:
                logger.error(f"Failed to load model {classifier_type.value}: {str(e)}")
                raise ModelNotFoundError(f"Failed to load model: {str(e)}")
    
    def get_model(self, classifier_type: ClassifierType):
        """Get a model, loading it if necessary."""
        self._load_model(classifier_type)
        return self._models[classifier_type]
    
    def get_input_shape(self, classifier_type: ClassifierType) -> Tuple[int, int]:
        """Get the input shape for a model."""
        self._load_model(classifier_type)
        return self._input_shapes[classifier_type]
    
    def is_model_available(self, classifier_type: ClassifierType) -> bool:
        """Check if a model file exists."""
        return os.path.exists(self._get_model_path(classifier_type))
    
    def get_model_info(self, classifier_type: ClassifierType) -> Dict[str, Any]:
        """Get detailed information about a model."""
        model_path = self._get_model_path(classifier_type)
        info = {
            'type': classifier_type.value,
            'path': model_path,
            'available': os.path.exists(model_path),
            'loaded': classifier_type in self._models,
        }
        
        if info['available']:
            info['file_size_mb'] = round(os.path.getsize(model_path) / (1024 * 1024), 2)
        
        if info['loaded']:
            info['input_shape'] = self._input_shapes[classifier_type]
            info['load_time_s'] = round(self._load_times.get(classifier_type, 0), 2)
        
        return info


# Global model loader instance (singleton)
model_loader = LazyModelLoader()


# ============================================================================
# DYNAMIC CLASS CONFIGURATION
# ============================================================================

def load_class_mapping(classifier_type: ClassifierType) -> Optional[Dict[str, int]]:
    """
    Load class mapping from JSON file created during training.
    Returns dict mapping class name -> index.
    """
    json_filename = f"{classifier_type.value}_classes.json"
    json_path = os.path.join(settings.BASE_DIR, 'models', json_filename)
    
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                class_indices = json.load(f)
            logger.info(f"Loaded class mapping from {json_path}: {class_indices}")
            return class_indices
        except Exception as e:
            logger.warning(f"Failed to load class mapping from {json_path}: {e}")
    
    return None


def get_dynamic_classes(classifier_type: ClassifierType) -> List[str]:
    """
    Get class list from trained model's JSON file or fall back to defaults.
    """
    class_mapping = load_class_mapping(classifier_type)
    
    if class_mapping:
        # Sort by index to get correct order
        sorted_classes = sorted(class_mapping.items(), key=lambda x: x[1])
        return [cls_name for cls_name, _ in sorted_classes]
    
    # Fall back to default configuration
    return CLASSIFIER_CONFIG[classifier_type].get('classes', [])


# ============================================================================
# MAIN CLASSIFICATION FUNCTION
# ============================================================================

def classify_image(
    image_path: str,
    classifier_type: ClassifierType,
    generate_heatmap: bool = False,
    use_tta: bool = True,
    assess_quality: bool = True,
    top_n: int = 3
) -> ClassificationResult:
    """
    Classify a medical image with comprehensive analysis.
    
    Args:
        image_path: Path to the image file
        classifier_type: Type of classification to perform
        generate_heatmap: Whether to generate Grad-CAM visualization
        use_tta: Whether to use Test-Time Augmentation
        assess_quality: Whether to perform image quality assessment
        top_n: Number of top predictions to return
        
    Returns:
        ClassificationResult with comprehensive prediction details
    """
    start_time = time.time()
    
    try:
        config = CLASSIFIER_CONFIG[classifier_type]
        classes = config['classes']
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ClassificationError(f"Failed to load image: {image_path}")
        
        # Calculate image hash for caching/tracking
        image_hash = hashlib.md5(img.tobytes()).hexdigest()[:12]
        
        # Assess image quality
        quality_assessment = None
        if assess_quality:
            quality_assessment = ImageQualityAnalyzer.assess_quality(img, classifier_type)
            if not quality_assessment.is_usable:
                raise ImageQualityError(
                    "Image quality is too poor for reliable analysis. " +
                    "; ".join(quality_assessment.suggestions)
                )
        
        # Get model
        model = model_loader.get_model(classifier_type)
        input_shape = model_loader.get_input_shape(classifier_type)
        
        # Special handling for brain classifier (uses VGG-16 preprocessing and cropping)
        if classifier_type == ClassifierType.BRAIN and BRAIN_CLASSIFIER_AVAILABLE:
            # Use brain_classifier module for specialized preprocessing
            class_idx, pred_probs = brain_classifier.classifier(image_path, model, input_shape)
            probabilities = pred_probs[0]  # Extract from batch dimension
            preprocessing_applied = ["brain_cropping", "vgg16_preprocessing"]
        else:
            # Standard preprocessing for other classifiers
            processed_img, preprocessing_applied = ImagePreprocessor.preprocess(img, classifier_type)
            
            # Perform prediction (with or without TTA)
            if use_tta:
                probabilities = TestTimeAugmentation.get_augmented_predictions(
                    model, processed_img, input_shape, classifier_type
                )
            else:
                resized = cv2.resize(processed_img, dsize=input_shape)
                img_expanded = np.expand_dims(resized, axis=0).astype(np.float32)
                img_normalized = img_expanded / 255.0
                predictions = model.predict(img_normalized, verbose=0)
                probabilities = predictions[0]
        
        # Handle binary classifiers (sigmoid output) vs multi-class (softmax)
        is_binary = config.get('is_binary', False)
        if is_binary and (len(probabilities) == 1 or np.isscalar(probabilities)):
            # Binary classifier: sigmoid output is P(positive class)
            p_positive = float(probabilities[0]) if hasattr(probabilities, '__getitem__') else float(probabilities)
            probabilities = np.array([1.0 - p_positive, p_positive])
        
        # Get results
        top_indices = np.argsort(probabilities)[::-1][:top_n]
        class_index = int(top_indices[0])
        confidence = float(probabilities[class_index])
        
        # Build top predictions list
        top_predictions = []
        for idx in top_indices:
            pred = PredictionDetail(
                class_name=classes[idx] if idx < len(classes) else f"Unknown ({idx})",
                probability=float(probabilities[idx]),
                severity=config['severity_mapping'].get(idx, Severity.MODERATE),
                description=config['descriptions'].get(idx, ""),
            )
            top_predictions.append(pred)
        
        # Get medical context
        severity = config['severity_mapping'].get(class_index, Severity.MODERATE)
        recommendations = config['recommendations'].get(class_index, ["Consult a medical professional"])
        urgency_level = config['urgency_mapping'].get(class_index, "Schedule appointment")
        
        # Add quality-based warnings
        if quality_assessment and quality_assessment.overall_quality == ImageQuality.POOR:
            recommendations = [
                "⚠️ Image quality is poor - consider retaking with better conditions"
            ] + recommendations
        
        # Add confidence warning
        confidence_threshold = config.get('confidence_threshold', 0.5)
        is_confident = confidence >= confidence_threshold
        
        if not is_confident:
            recommendations = [
                "⚠️ Low confidence prediction - results may be unreliable",
                "Consider retaking the image with better quality or lighting"
            ] + recommendations
        
        # Generate heatmap if requested
        heatmap_base64 = None
        if generate_heatmap:
            try:
                resized = cv2.resize(processed_img, dsize=input_shape)
                img_expanded = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0
                heatmap = GradCAMGenerator.generate_heatmap(model, img_expanded, class_index)
                
                if heatmap is not None:
                    img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    overlay = GradCAMGenerator.overlay_heatmap(img_rgb, heatmap)
                    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
            except Exception as e:
                logger.warning(f"Failed to generate heatmap: {str(e)}")
        
        # Build medical info
        medical_info = {
            'condition': classes[class_index],
            'description': config['descriptions'].get(class_index, ""),
            'urgency': urgency_level,
            'specialist': _get_specialist_for_classifier(classifier_type),
        }
        
        inference_time = (time.time() - start_time) * 1000
        
        return ClassificationResult(
            class_index=class_index,
            class_name=classes[class_index],
            confidence=confidence,
            severity=severity,
            is_confident=is_confident,
            top_predictions=top_predictions,
            all_probabilities=[float(p) for p in probabilities],
            recommendations=recommendations.copy(),
            medical_info=medical_info,
            urgency_level=urgency_level,
            preprocessing_applied=preprocessing_applied,
            inference_time_ms=inference_time,
            model_version="2.0",
            tta_enabled=use_tta,
            heatmap_base64=heatmap_base64,
            quality_assessment=quality_assessment,
            image_hash=image_hash,
        )
        
    except (ModelNotFoundError, ImageValidationError, ImageQualityError):
        raise
    except Exception as e:
        logger.error(f"Classification error for {classifier_type.value}: {str(e)}")
        raise ClassificationError(f"Failed to classify image: {str(e)}")


def _get_specialist_for_classifier(classifier_type: ClassifierType) -> str:
    """Get the recommended specialist type for each classifier."""
    specialists = {
<<<<<<< HEAD
=======
        ClassifierType.SKIN: "Dermatologist",
>>>>>>> origin/main
        ClassifierType.BRAIN: "Neurologist/Neuro-oncologist",
        ClassifierType.BONES: "Orthopedic Specialist/Radiologist",
    }
    return specialists.get(classifier_type, "Medical Professional")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_models_status() -> Dict[str, Any]:
    """Check availability status of all models."""
    status = {}
    for classifier_type in ClassifierType:
        status[classifier_type.value] = model_loader.get_model_info(classifier_type)
        status[classifier_type.value]['classes'] = CLASSIFIER_CONFIG[classifier_type]['classes']
    return status


def get_classifier_info(classifier_type: ClassifierType) -> Dict[str, Any]:
    """Get comprehensive information about a classifier."""
    config = CLASSIFIER_CONFIG[classifier_type]
    return {
        'type': classifier_type.value,
        'available': model_loader.is_model_available(classifier_type),
        'classes': config['classes'],
        'num_classes': len(config['classes']),
        'confidence_threshold': config.get('confidence_threshold', 0.5),
        'severity_levels': {
            cls: config['severity_mapping'].get(idx, Severity.MODERATE).value
            for idx, cls in enumerate(config['classes'])
        },
        'descriptions': config.get('descriptions', {}),
        'preprocessing_steps': config.get('preprocessing', []),
        'tta_enabled': len(config.get('tta_rotations', [])) > 1 or len(config.get('tta_flips', [])) > 1,
        'specialist': _get_specialist_for_classifier(classifier_type),
    }


def get_model_info(classifier_type: ClassifierType) -> Dict[str, Any]:
    """Get detailed model information."""
    return model_loader.get_model_info(classifier_type)


def get_all_conditions(classifier_type: ClassifierType) -> List[Dict[str, Any]]:
    """Get information about all detectable conditions for a classifier."""
    config = CLASSIFIER_CONFIG[classifier_type]
    return [
        {
            'name': cls,
            'severity': config['severity_mapping'].get(idx, Severity.MODERATE).value,
            'description': config['descriptions'].get(idx, ""),
            'urgency': config['urgency_mapping'].get(idx, ""),
        }
        for idx, cls in enumerate(config['classes'])
    ]
