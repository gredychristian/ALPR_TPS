import os
import csv
import cv2
import numpy as np


def init_csv(log_file):
    """
    Inisialisasi file CSV untuk logging
    """
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "date",
                    "time",
                    "plate_number",
                    "vehicle_type",
                    "color",
                    "confidence",
                    "filename",
                ]
            )


def save_frame(frame, path):
    """
    Simpan frame ke file
    """
    try:
        cv2.imwrite(path, frame)
        return True
    except Exception as e:
        print(f"❌ Gagal menyimpan frame: {e}")
        return False


def append_log(log_file, row):
    """
    Tambahkan data ke log CSV
    """
    try:
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        return True
    except Exception as e:
        print(f"❌ Gagal menulis log: {e}")
        return False


def resize_keep_aspect(img, target_w, target_h):
    """
    Resize image dengan menjaga aspect ratio
    """
    if img is None or img.size == 0:
        return np.full((target_h, target_w, 3), 255, dtype=np.uint8)

    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create canvas dengan background putih
    canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)

    # Center the image
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2

    # Pastikan indeks tidak melebihi batas canvas
    end_y = min(y_off + new_h, target_h)
    end_x = min(x_off + new_w, target_w)
    actual_h = end_y - y_off
    actual_w = end_x - x_off

    if actual_h > 0 and actual_w > 0:
        canvas[y_off:end_y, x_off:end_x] = resized[:actual_h, :actual_w]

    return canvas
