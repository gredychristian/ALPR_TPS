import os

# Direktori untuk menyimpan hasil capture
CAPTURE_DIR = "capture_gatein"
os.makedirs(CAPTURE_DIR, exist_ok=True)

# File log untuk mencatat hasil capture
LOG_FILE = os.path.join(CAPTURE_DIR, "log_gatein.csv")

# Model YOLO
YOLO_MODEL_PATH = "weights/best.pt"

# Video source (0 = webcam, atau path ke file video)
MEDIA_SOURCE = 0

# OCR Settings
OCR_LANGUAGES = ["en"]
OCR_GPU = False

# Confidence threshold untuk deteksi
CONFIDENCE_THRESHOLD = 0.25

# OCR interval (setiap n frame)
OCR_INTERVAL = 15
