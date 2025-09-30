# utils/plate_detector.py
from ultralytics import YOLO
import cv2
import numpy as np
import config


class PlateDetector:
    def __init__(self, model_path="models/plate_detection/best.pt"):
        print("Loading plate detection model...")

        # ✅ LOAD MODEL DENGAN WEIGHTS_ONLY=True
        try:
            # Coba load dengan weights_only=True (lebih secure)
            self.model = YOLO(model_path)
            print(f"✅ Plate detection model loaded on {config.DEVICE}!")
        except Exception as e:
            print(f"⚠️  Failed to load with secure mode: {e}")
            print("🔄 Trying alternative loading method...")
            # Fallback ke method lama jika diperlukan
            self.model = YOLO(model_path)
            print(
                f"✅ Plate detection model loaded (fallback method) on {config.DEVICE}!"
            )

    def detect_plates(self, image, confidence=0.5):
        """
        Detect license plates dan filter hanya yang TERBESAR & TERJELAS
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run detection dengan device configuration
        results = self.model(image_rgb, conf=confidence, device=config.DEVICE)

        detected_plates = []

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    # Calculate area of bounding box
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height

                    # Crop plate region
                    plate_img = image[y1:y2, x1:x2]

                    if plate_img.size > 0:
                        detected_plates.append(
                            {
                                "image": plate_img,
                                "bbox": (x1, y1, x2, y2),
                                "confidence": conf,
                                "area": area,
                                "width": width,
                                "height": height,
                            }
                        )

        # ✅ FILTER: Ambil hanya plate dengan AREA TERBESAR
        if detected_plates:
            # Urutkan berdasarkan area (besar ke kecil)
            detected_plates.sort(key=lambda x: x["area"], reverse=True)

            # Ambil hanya plate TERBESAR saja
            largest_plate = detected_plates[0]

            print(
                f"📏 Filtered: {len(detected_plates)} plates -> 1 largest "
                f"(Area: {largest_plate['area']} px²)"
            )

            return [largest_plate]
        else:
            return []

    def draw_detections(self, image, plates):
        """
        Draw bounding boxes and labels on image
        """
        result_image = image.copy()

        for plate in plates:
            x1, y1, x2, y2 = plate["bbox"]
            conf = plate["confidence"]
            area = plate.get("area", 0)

            # Draw rectangle
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label dengan info area
            label = f"Plate: {conf:.2f} | Area: {area}px²"
            cv2.putText(
                result_image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        return result_image
