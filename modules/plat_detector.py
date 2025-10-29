import cv2
import numpy as np
from ultralytics import YOLO


class PlatDetector:
    def __init__(self, model_path, conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, image):
        """Deteksi plat nomor dalam gambar"""
        try:
            results = self.model.predict(image, conf=self.conf_threshold, iou=0.5)

            if len(results[0].boxes) == 0:
                return None, None

            # Ambil bounding box plat pertama (terbesar)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            if len(boxes) > 0:
                x1, y1, x2, y2 = boxes[0].astype(int)

                # Pastikan koordinat dalam range image
                h, w = image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Pastikan crop valid
                if x2 > x1 and y2 > y1:
                    cropped_plate = image[y1:y2, x1:x2]
                    return cropped_plate, (x1, y1, x2, y2)

            return None, None

        except Exception as e:
            # print(f"❌ Plate detection error: {str(e)}")
            return None, None

    def draw_detection(self, image, bbox):
        """Gambar bounding box plat pada gambar (BGR format)"""
        if bbox:
            x1, y1, x2, y2 = bbox
            # Warna hijau untuk BGR: (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        return image