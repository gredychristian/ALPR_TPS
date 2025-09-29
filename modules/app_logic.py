import cv2
import easyocr
import datetime
import os
import numpy as np
from modules.utils_io import init_csv, save_frame, append_log
from modules.ocr_preprocessing import process_ocr_advanced
from modules.color_detection import detect_plate_color
from modules.config import CAPTURE_DIR, LOG_FILE
from modules.ui_layout import create_dashboard


def improved_plate_filtering(detections, frame_shape):
    """
    Filter deteksi yang lebih baik berdasarkan paper Light-Edge
    """
    filtered_detections = []
    img_height, img_width = frame_shape[:2]

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        width = x2 - x1
        height = y2 - y1

        # Aspect ratio filtering dari paper (1.0 - 8.0)
        aspect_ratio = width / height if height > 0 else 0
        if not (1.0 <= aspect_ratio <= 8.0):
            continue

        # Size filtering - plat harus reasonable size
        min_width = img_width * 0.05  # 5% dari lebar frame
        max_width = img_width * 0.4  # 40% dari lebar frame
        min_height = img_height * 0.03  # 3% dari tinggi frame
        max_height = img_height * 0.3  # 30% dari tinggi frame

        if not (min_width <= width <= max_width and min_height <= height <= max_height):
            continue

        # Area coverage filtering
        detection_area = width * height
        frame_area = img_width * img_height
        area_ratio = detection_area / frame_area

        if not (0.001 <= area_ratio <= 0.2):  # 0.1% sampai 20% frame
            continue

        # Confidence threshold
        if detection["confidence"] < 0.5:
            continue

        filtered_detections.append(detection)

    return filtered_detections


def run_app(model, cap):
    """
    Fungsi utama untuk menjalankan aplikasi deteksi plat nomor dengan UI dashboard
    """
    # Initialize EasyOCR
    reader = easyocr.Reader(["en"], gpu=False)

    # Initialize CSV log
    init_csv(LOG_FILE)

    frame_count = 0

    # Variabel untuk menyimpan data capture terakhir
    last_capture_data = {
        "capture_time": "",
        "plate_image": None,
        "plate_number": "BELUM ADA CAPTURE",
        "vehicle_type": "BELUM ADA CAPTURE",
        "raw_image": None,
        "confidence": 0.0,
    }

    # Buat window dengan properties yang diinginkan
    window_name = "ALPR System - PT TPS Surabaya"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Set ukuran minimum window
    cv2.resizeWindow(window_name, 1200, 700)

    print("🚀 Memulai Aplikasi Deteksi Plat Nomor PT TPS Surabaya")
    print("📋 Tekan 'C' untuk capture, 'ESC' untuk keluar")
    print("💡 Menggunakan teknik Light-Edge dari penelitian terbaru")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ Gagal membaca frame")
            break

        frame_count += 1
        display_frame = frame.copy()

        # YOLO detection dengan confidence threshold reasonable
        results = model(frame, verbose=False, conf=0.5)

        plates_detected = []
        current_plates_info = []

        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy()

                for box, conf, cls in zip(boxes, confs, clss):
                    x1, y1, x2, y2 = map(int, box)
                    width = x2 - x1
                    height = y2 - y1

                    # Basic filtering
                    aspect_ratio = width / height if height > 0 else 0

                    if (
                        aspect_ratio < 1.0
                        or aspect_ratio > 8.0
                        or width < 40
                        or height < 15
                    ):
                        continue

                    roi = frame[y1:y2, x1:x2]

                    if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 40:
                        continue

                    plates_detected.append(
                        {
                            "bbox": (x1, y1, x2, y2),
                            "roi": roi,
                            "confidence": conf,
                        }
                    )

        # Apply improved filtering dari paper Light-Edge
        plates_detected = improved_plate_filtering(plates_detected, frame.shape)

        # Gambar bounding box dan update info untuk yang terfilter
        for plate in plates_detected:
            x1, y1, x2, y2 = plate["bbox"]

            # Gambar bounding box untuk live preview
            color = (0, 255, 0)  # Hijau untuk plat terdeteksi
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                display_frame,
                f"PLATE {plate['confidence']:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

            current_plates_info.append(
                {"bbox": (x1, y1, x2, y2), "confidence": plate["confidence"]}
            )

        # Buat dashboard UI
        dashboard = create_dashboard(
            display_frame, last_capture_data, current_plates_info
        )

        # Tampilkan dashboard
        cv2.imshow(window_name, dashboard)

        # Handle keyboard input dan window events
        key = cv2.waitKey(1) & 0xFF

        # ESC key untuk keluar
        if key == 27:  # ESC key
            print("👋 Keluar dari aplikasi...")
            break
        # C key untuk capture
        elif key == ord("c") and plates_detected:
            # Capture frame dan proses plat yang terdeteksi
            timestamp = datetime.datetime.now()
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")
            filename = f"capture_{timestamp_str}.jpg"
            filepath = os.path.join(CAPTURE_DIR, filename)

            # Ambil plat dengan confidence tertinggi
            best_plate = max(plates_detected, key=lambda x: x["confidence"])

            try:
                print(
                    f"🔍 Processing plate with confidence: {best_plate['confidence']:.2f}"
                )

                # Process OCR advanced dengan teknik Light-Edge
                plate_number, confidence = process_ocr_advanced(
                    best_plate["roi"], reader
                )

                # Deteksi warna plat
                color_name, vehicle_type = detect_plate_color(best_plate["roi"])

                # Update last capture data
                last_capture_data = {
                    "capture_time": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "plate_image": best_plate["roi"],
                    "plate_number": plate_number,
                    "vehicle_type": vehicle_type,
                    "raw_image": frame.copy(),
                    "confidence": confidence,
                }

                # Save gambar asli (dengan bounding box)
                cv2.imwrite(filepath, frame)

                # Log ke CSV
                append_log(
                    LOG_FILE,
                    [
                        timestamp.strftime("%Y-%m-%d"),
                        timestamp.strftime("%H:%M:%S"),
                        plate_number,
                        vehicle_type,
                        color_name,
                        f"{confidence:.2f}",
                        filename,
                    ],
                )

                print(
                    f"✅ Capture berhasil: {plate_number} - {vehicle_type} (Conf: {confidence:.2f})"
                )
                print(f"💾 Disimpan: {filepath}")

            except Exception as e:
                print(f"❌ Error saat processing capture: {e}")
                last_capture_data = {
                    "capture_time": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "plate_image": best_plate["roi"],
                    "plate_number": "ERROR",
                    "vehicle_type": "ERROR",
                    "raw_image": frame.copy(),
                    "confidence": 0.0,
                }

        elif key == ord("c") and not plates_detected:
            print("⚠️  Tidak ada plat yang terdeteksi untuk dicapture")

        # Check jika window ditutup dengan tombol silang
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("👋 Window ditutup dengan tombol silang")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Aplikasi ditutup dengan aman")
