import cv2
import re
import numpy as np
import easyocr

def simple_ocr_processing(img, reader):
    """
    OCR sederhana tapi efektif - kembali ke basic
    """
    print("🔍 Simple OCR processing...")
    
    if img is None or img.size == 0:
        return "NO IMAGE", 0.0
    
    try:
        # Preprocessing sederhana
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Resize jika terlalu kecil
        height, width = gray.shape
        if height < 50:
            scale = 80 / height
            new_width = int(width * scale)
            gray = cv2.resize(gray, (new_width, 80), interpolation=cv2.INTER_CUBIC)
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # OCR dengan parameter standard
        results = reader.readtext(
            enhanced,
            detail=1,
            text_threshold=0.4,
            low_text=0.4,
            link_threshold=0.4
        )
        
        # Kumpulkan semua teks
        all_texts = []
        total_conf = 0.0
        
        for bbox, text, conf in results:
            clean_text = str(text).strip().upper()
            clean_text = re.sub(r'[^A-Z0-9]', '', clean_text)
            
            if len(clean_text) >= 1 and conf > 0.3:
                all_texts.append(clean_text)
                total_conf += conf
        
        if not all_texts:
            return "TIDAK TERBACA", 0.0
        
        # Gabungkan semua teks
        combined = ''.join(all_texts)
        avg_conf = total_conf / len(all_texts)
        
        print(f"✅ Simple OCR: {combined} (Conf: {avg_conf:.2f})")
        return combined, avg_conf
        
    except Exception as e:
        print(f"❌ Simple OCR Error: {e}")
        return "ERROR", 0.0

# Tetap simpan function advanced untuk pilihan
def process_ocr_advanced(img, reader):
    """
    Fallback ke simple OCR dulu untuk testing
    """
    return simple_ocr_processing(img, reader)