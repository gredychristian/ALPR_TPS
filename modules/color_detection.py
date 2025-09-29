import cv2
import numpy as np


def detect_plate_color(plate_img):
    """
    Deteksi warna dominan pada plat nomor.
    Return: (color_key, vehicle_type_label)
    """
    if plate_img is None or plate_img.size == 0:
        return "lain", "TIDAK TERDEFINISI"

    try:
        # Convert to HSV
        hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)

        # Preprocessing - noise reduction
        hsv = cv2.medianBlur(hsv, 5)

        # Definisikan range warna untuk plat Indonesia
        mask_white = cv2.inRange(hsv, (0, 0, 180), (180, 50, 255))  # Plat putih
        mask_black = cv2.inRange(hsv, (0, 0, 0), (180, 255, 80))  # Plat hitam
        mask_yellow = cv2.inRange(hsv, (20, 80, 80), (40, 255, 255))  # Plat kuning
        mask_red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))  # Plat merah
        mask_red2 = cv2.inRange(
            hsv, (160, 70, 50), (180, 255, 255)
        )  # Plat merah (range kedua)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_green = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))  # Plat hijau

        total_pixels = float(plate_img.shape[0] * plate_img.shape[1]) + 1e-9

        # Hitung persentase setiap warna
        ratios = {
            "putih": np.sum(mask_white > 0) / total_pixels,
            "hitam": np.sum(mask_black > 0) / total_pixels,
            "kuning": np.sum(mask_yellow > 0) / total_pixels,
            "merah": np.sum(mask_red > 0) / total_pixels,
            "hijau": np.sum(mask_green > 0) / total_pixels,
        }

        # Cari warna dominan
        dominant_color = max(ratios, key=ratios.get)

        # Threshold minimum untuk dianggap valid
        if ratios[dominant_color] < 0.05:  # 5% threshold
            dominant_color = "lain"

        # Mapping warna ke jenis kendaraan (sesuai aturan Indonesia)
        color_mapping = {
            "putih": "KENDARAAN PRIBADI / BADAN HUKUM",
            "hitam": "KENDARAAN PRIBADI (PLAT HITAM)",
            "kuning": "UMUM / TRANSPORTASI PUBLIK",
            "merah": "INSTANSI PEMERINTAH",
            "hijau": "PERDAGANGAN BEBAS / DIPLOMATIK",
            "lain": "TIDAK TERDEFINISI",
        }

        return dominant_color, color_mapping[dominant_color]

    except Exception:
        return "lain", "TIDAK TERDEFINISI"
