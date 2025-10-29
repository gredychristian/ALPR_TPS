import cv2
import numpy as np
import random
from ultralytics import YOLO


class CharDetector:
    def __init__(
        self, model_path, conf_threshold=0.3, iou_threshold=0.4
    ):  # <- TAMBAH PARAMETER INI
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold  # <- SIMPAN SEBAGAI ATTRIBUTE

    def detect(self, plate_image):
        """Deteksi karakter dalam gambar plat"""
        try:

            # Pastikan image dalam format yang benar untuk YOLO
            if len(plate_image.shape) == 2:  # Grayscale
                # Convert ke 3 channel
                plate_image = cv2.cvtColor(plate_image, cv2.COLOR_GRAY2RGB)
            elif plate_image.shape[2] == 1:  # Single channel
                plate_image = cv2.cvtColor(plate_image, cv2.COLOR_GRAY2RGB)

            # print(f"🔍 Char detection - After conversion: {plate_image.shape}")

            # GUNAKAN iou_threshold di sini
            results = self.model.predict(
                plate_image, conf=self.conf_threshold, iou=self.iou_threshold
            )

            characters = []
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                # Urutkan dari kiri ke kanan
                sorted_indices = boxes[:, 0].argsort()

                for idx in sorted_indices:
                    x1, y1, x2, y2 = boxes[idx].astype(int)
                    char_class = results[0].names[class_ids[idx]]
                    characters.append({"char": char_class, "bbox": (x1, y1, x2, y2)})

            # print(f"🔍 Characters detected: {len(characters)}")
            return characters

        except Exception as e:
            # print(f"❌ Character detection error: {str(e)}")
            return []

    def draw_detections(self, image, characters):
        """Gambar bounding box dan label karakter dengan warna YOLO otomatis"""
        try:
            # Pastikan image dalam format RGB untuk drawing
            if len(image.shape) == 2:  # Grayscale
                output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                output_image = image.copy()

            for char_data in characters:
                x1, y1, x2, y2 = char_data["bbox"]
                char_text = char_data["char"]

                # Gunakan warna YOLO otomatis berdasarkan class name
                # YOLO generate color dari hash class name
                color = self._generate_color(char_text)

                # Draw bounding box dengan warna YOLO
                cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)

                # Draw label dengan warna yang sama
                cv2.putText(
                    output_image,
                    char_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )

            return output_image  # Kembalikan dalam format RGB

        except Exception as e:
            # print(f"❌ Drawing error: {str(e)}")
            return image

    def _generate_color(self, class_name):
        """Generate warna konsisten berdasarkan class name (seperti YOLO)"""
        # Hash class name untuk mendapatkan nilai konsisten
        hash_val = hash(class_name) % (256**3)

        # Extract RGB values dari hash
        r = (hash_val // (256**2)) % 256
        g = (hash_val // 256) % 256
        b = hash_val % 256

        return (int(r), int(g), int(b))

    def get_plate_text(self, characters):
        """Gabungkan karakter menjadi teks plat"""
        return "".join([char["char"] for char in characters])
