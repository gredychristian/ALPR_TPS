import cv2
from ultralytics import YOLO
from modules.app_logic import run_app
from modules.config import MEDIA_SOURCE


def main():
    model_path = "weights/best.pt"

    # Load model dengan verbose=False
    model = YOLO(model_path)
    cap = cv2.VideoCapture(MEDIA_SOURCE)

    if not cap.isOpened():
        print("❌ Video/kamera gagal dibuka")
        return

    print("🚀 Memulai Aplikasi Deteksi Plat Nomor PT TPS Surabaya")
    run_app(model, cap)


if __name__ == "__main__":
    main()
