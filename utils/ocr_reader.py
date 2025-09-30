# utils/ocr_reader.py
import easyocr
import cv2
import numpy as np
import re
from collections import Counter
import config


class OCRReader:
    def __init__(self):
        print("Loading EasyOCR model...")

        # ✅ AUTO GPU/CPU UNTUK EASYOCR
        gpu = config.DEVICE == "cuda"

        self.reader = easyocr.Reader(["en"], gpu=gpu, download_enabled=True)

        device_status = "GPU" if gpu else "CPU"
        print(f"✅ EasyOCR model loaded on {device_status}!")

    def clean_plate_text(self, text):
        """Cleaning function untuk plat nomor"""
        cleaned = "".join(c for c in text if c.isalnum())
        cleaned = cleaned.upper()

        false_positives = ["LICENSE", "PLATE", "NUMBER", "INDONESIA", "IDN"]
        for fp in false_positives:
            cleaned = cleaned.replace(fp, "")

        if not cleaned:
            cleaned = "".join(c for c in text if c.isalnum()).upper()

        print(f"🔧 Text Cleaning: '{text}' -> '{cleaned}'")
        return cleaned

    def analyze_text_clusters(self, results):
        """
        ANALISIS CLUSTER: Kelompokkan text berdasarkan size & position
        """
        if not results:
            return []

        text_elements = []
        for bbox, text, confidence in results:
            points = np.array(bbox)
            height = np.max(points[:, 1]) - np.min(points[:, 1])
            width = np.max(points[:, 0]) - np.min(points[:, 0])
            center_y = np.mean(points[:, 1])

            text_elements.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "height": height,
                    "width": width,
                    "center_y": center_y,
                    "bbox": bbox,
                }
            )

        # Debug: print semua yang terdeteksi
        print("📝 ALL DETECTED TEXTS:")
        for i, elem in enumerate(text_elements):
            print(
                f"   {i+1}. '{elem['text']}' (H:{elem['height']:.1f}px, W:{elem['width']:.1f}px, Y:{elem['center_y']:.1f})"
            )

        # Kelompokkan berdasarkan height (cluster analysis)
        heights = [elem["height"] for elem in text_elements]
        if len(heights) >= 2:
            sorted_heights = sorted(heights)
            gaps = [
                sorted_heights[i + 1] - sorted_heights[i]
                for i in range(len(sorted_heights) - 1)
            ]

            if gaps:
                max_gap_idx = gaps.index(max(gaps))
                threshold = (
                    sorted_heights[max_gap_idx] + sorted_heights[max_gap_idx + 1]
                ) / 2

                print(f"📊 Height Analysis: Threshold = {threshold:.1f}px")

                large_texts = [
                    elem for elem in text_elements if elem["height"] >= threshold
                ]

                if large_texts:
                    large_texts.sort(key=lambda x: x["confidence"], reverse=True)
                    selected = large_texts[0]
                    print(
                        f"🎯 SELECTED: '{selected['text']}' (Main plate - largest font)"
                    )
                    return [selected]

        text_elements.sort(key=lambda x: x["confidence"], reverse=True)
        if text_elements:
            print(
                f"🎯 SELECTED: '{text_elements[0]['text']}' (Fallback - highest confidence)"
            )
            return [text_elements[0]]

        return []

    def filter_by_plausible_plate_format(self, text_elements):
        """
        FILTER FORMAT: Prioritaskan text yang mirip format plat nomor
        """

        def plate_likelihood(text):
            clean_text = self.clean_plate_text(text)

            patterns = [
                r"^[A-Z]{1,2}\d{1,4}[A-Z]{1,3}$",
                r"^[A-Z]{1,3}\d{1,4}$",
                r"^\d{1,4}[A-Z]{1,3}$",
                r"^[A-Z]{2,3}\d{2,4}$",
            ]

            for pattern in patterns:
                if re.match(pattern, clean_text):
                    return 2.0

            if len(clean_text) >= 4 and len(clean_text) <= 10:
                char_count = sum(1 for c in clean_text if c.isalpha())
                digit_count = sum(1 for c in clean_text if c.isdigit())

                if char_count >= 1 and digit_count >= 1:
                    return 1.5
                elif char_count >= 2 or digit_count >= 2:
                    return 1.0

            return 0.1

        scored_elements = []
        for elem in text_elements:
            score = plate_likelihood(elem["text"])
            elem["plate_score"] = score
            scored_elements.append(elem)

        print("📋 PLATE FORMAT SCORES:")
        for elem in scored_elements:
            print(f"   '{elem['text']}': Score {elem['plate_score']:.1f}")

        if scored_elements and scored_elements[0]["plate_score"] > 0.5:
            return [scored_elements[0]]

        return scored_elements

    def smart_text_selection(self, results):
        """
        SELECTION SMART: Kombinasi size analysis + format matching
        """
        if not results:
            return []

        size_filtered = self.analyze_text_clusters(results)

        if size_filtered:
            format_filtered = self.filter_by_plausible_plate_format(size_filtered)

            if format_filtered:
                selected = format_filtered[0]
                print(
                    f"🏆 FINAL SELECTION: '{selected['text']}' "
                    f"(H:{selected['height']:.1f}px, Score:{selected.get('plate_score', 0):.1f})"
                )
                return format_filtered

        results.sort(key=lambda x: x[2], reverse=True)
        fallback = [
            {
                "text": results[0][1],
                "confidence": results[0][2],
                "height": 0,
                "plate_score": 0,
            }
        ]
        print(f"🔄 FALLBACK: '{fallback[0]['text']}' (Highest confidence)")
        return fallback

    def preprocess_plate(self, plate_image):
        """
        Simple preprocessing untuk plat nomor
        """
        # Convert to grayscale
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image

        # Resize jika terlalu kecil
        h, w = gray.shape
        if h < 40:
            scale = 60 / h
            new_w = int(w * scale)
            gray = cv2.resize(gray, (new_w, 60))

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        return enhanced

    def read_text(self, plate_image):
        try:
            import config

            if plate_image.size == 0:
                return "", 0.0

            processed_plate = self.preprocess_plate(plate_image)

            # OCR detection
            results = self.reader.readtext(
                processed_plate,
                detail=1,
                paragraph=False,
                text_threshold=0.3,
                low_text=0.2,
                width_ths=0.5,
                height_ths=0.5,
            )

            if results:
                # Smart selection
                filtered_results = self.smart_text_selection(results)

                if filtered_results:
                    best_match = filtered_results[0]
                    cleaned_text = self.clean_plate_text(best_match["text"])
                    confidence = best_match["confidence"]

                    if len(cleaned_text) >= config.MIN_PLATE_TEXT_LENGTH:
                        return cleaned_text, confidence
                    else:
                        print(f"⚠️  Rejected: '{cleaned_text}' (too short)")

            return "", 0.0

        except Exception as e:
            print(f"OCR Error: {e}")
            return "", 0.0
