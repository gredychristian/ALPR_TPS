from ultralytics import YOLO


def load_yolo_model(model_path: str):
    """
    Load model YOLO dari file .pt
    """
    try:
        model = YOLO(model_path)
        print(f"✅ Model YOLO loaded: {model_path}")
        return model
    except Exception as e:
        print(f"❌ Gagal load model YOLO: {e}")
        return None


def detect_plates(model, image, confidence_threshold=0.5):
    """
    Deteksi plat nomor menggunakan YOLO
    """
    if model is None or image is None:
        return []

    try:
        results = model(image)
        detections = []

        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy()

                for box, conf, cls in zip(boxes, confs, clss):
                    if conf >= confidence_threshold:
                        detections.append(
                            {"bbox": box, "confidence": conf, "class": cls}
                        )

        return detections

    except Exception as e:
        print(f"❌ Error dalam deteksi plat: {e}")
        return []
