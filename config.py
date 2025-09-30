# config.py
import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# ✅ AUTO GPU/CPU DETECTION
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🎯 Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")

# Model configurations
PLATE_DETECTOR_MODEL = "models/plate_detection/best.pt"
OCR_MODEL_TYPE = "easyocr"

# Confidence thresholds
DETECTION_CONFIDENCE_THRESHOLD = 0.6
OCR_CONFIDENCE_THRESHOLD = 0.3
MIN_PLATE_TEXT_LENGTH = 3

# ✅ BACKWARD COMPATIBILITY
CONFIDENCE_THRESHOLD = DETECTION_CONFIDENCE_THRESHOLD

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
