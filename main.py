"""
ALPR_TPS - Gate IN (final)
Gabungan cell 1..5 dengan:
- UI layout mirip mockup (kiri: preview + info, kanan: video)
- Aspect-ratio-preserving resize untuk video & preview
- OCR postprocessing (cell 5) -> ambil ANGKA utama
- Deteksi warna dasar plat -> map ke jenis kendaraan
- Simpan foto ke `capture_gatein` dan log ke `log_gatein.csv` saat trigger Capture (C)
"""

import os
import csv
import re
import cv2
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from datetime import datetime
from ultralytics import YOLO
import easyocr
import numpy as np
import matplotlib.patches as patches  # optional for debugging/visual (not used in UI)

# ---------------- CONFIG ----------------
VIDEO_SOURCE = 0  # 0 = webcam, or path "Traffic_Light.mp4"
MODEL_PATH = "weights/best.pt"  # ganti sesuai lokasi model Anda
CAPTURE_DIR = "capture_gatein"
LOG_FILE = "capture_gatein/log_gatein.csv"

os.makedirs(CAPTURE_DIR, exist_ok=True)

# Inisialisasi CSV bila belum ada
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "time", "plate_number", "vehicle_type", "filename"])

# ---------------- Load models ----------------
try:
    yolo_model = YOLO(MODEL_PATH)
except Exception as e:
    print("Gagal load YOLO model:", e)
    raise

try:
    reader = easyocr.Reader(["en"], gpu=False)  # set gpu=True jika tersedia dan ingin pakai GPU
except Exception as e:
    print("Gagal inisialisasi EasyOCR:", e)
    raise

# ---------------- Helper functions (cell5 style) ----------------
def bbox_metrics(bbox):
    xs = [float(p[0]) for p in bbox]
    ys = [float(p[1]) for p in bbox]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)
    area = w * h
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    return x_min, y_min, x_max, y_max, w, h, area, cx, cy

def numeric_substrings(s):
    return re.findall(r"\d+", s)

def detect_plate_color(plate_img):
    """
    Deteksi warna dasar plat sederhana berbasis HSV area rata-rata.
    Mengembalikan (color_key, vehicle_type_label)
    color_key in {"putih","hitam","kuning","merah","hijau","lain"}
    mapping ke jenis kendaraan sesuai Peraturan (disesuaikan)
    """
    if plate_img is None or plate_img.size == 0:
        return "lain", "TIDAK TERDEFINISI"

    hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
    # Ambil rasio piksel tiap mask
    mask_white = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
    mask_black = cv2.inRange(hsv, (0, 0, 0), (180, 255, 60))
    mask_yellow = cv2.inRange(hsv, (15, 80, 100), (40, 255, 255))
    mask_red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    mask_red2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_green = cv2.inRange(hsv, (35, 60, 50), (90, 255, 255))

    total = plate_img.shape[0] * plate_img.shape[1] + 1e-9
    ratios = {
        "putih": np.sum(mask_white > 0) / total,
        "hitam": np.sum(mask_black > 0) / total,
        "kuning": np.sum(mask_yellow > 0) / total,
        "merah": np.sum(mask_red > 0) / total,
        "hijau": np.sum(mask_green > 0) / total,
    }
    color = max(ratios, key=ratios.get)
    # threshold: jika rasio tertinggi sangat kecil, treat as 'lain'
    if ratios[color] < 0.02:
        color = "lain"

    mapping = {
        "putih": "KENDARAAN PRIBADI / BADAN HUKUM",
        "hitam": "KENDARAAN PRIBADI (PLAT HITAM)",
        "kuning": "UMUM / TRANSPORTASI PUBLIK",
        "merah": "INSTANSI PEMERINTAH",
        "hijau": "PERDAGANGAN BEBAS",
        "lain": "TIDAK TERDEFINISI",
    }
    return color, mapping[color]

def process_ocr_and_pick_number(img_full, plate_img):
    """
    Jalankan EasyOCR pada plate_img (atau img_full jika diperlukan),
    lalu pakai scoring seperti cell5 untuk ambil angka utama (2..4 digit prior).
    Mengembalikan (main_number (str) or 'UNKNOWN', vehicle_type_label)
    """
    # Use plate_img as input for OCR if available (clearer), else full
    ocr_input = plate_img if plate_img is not None else img_full
    if ocr_input is None:
        return "UNKNOWN", "TIDAK TERDEFINISI"

    # EasyOCR mengharapkan BGR/ndarray atau path; kita beri ndarray
    try:
        results = reader.readtext(ocr_input)
    except Exception as e:
        # fallback: return unknown
        print("OCR error:", e)
        return "UNKNOWN", "TIDAK TERDEFINISI"

    # results entries may be (bbox, text, prob) or other shapes; handle robustly
    parsed = []
    for r in results:
        if len(r) == 3:
            bbox, text, prob = r
        elif len(r) == 2:
            bbox, text = r
            prob = 1.0
        else:
            # unexpected, try to coerce
            try:
                bbox = r[0]
                text = r[1]
                prob = float(r[2]) if len(r) > 2 else 1.0
            except:
                continue
        parsed.append((bbox, str(text), float(prob)))

    h_img, w_img = ocr_input.shape[:2]
    center_x_img = w_img / 2.0

    candidates = []
    for bbox, text, prob in parsed:
        raw_text = str(text)
        cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_text).upper()
        nums = numeric_substrings(cleaned)
        if len(nums) == 0:
            continue
        num_sub = max(nums, key=len)
        try:
            x_min, y_min, x_max, y_max, w_box, h_box, area, cx, cy = bbox_metrics(bbox)
        except Exception:
            # fallback approximate bbox center
            xs = [float(p[0]) for p in bbox]
            ys = [float(p[1]) for p in bbox]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            w_box = max(1.0, x_max - x_min)
            h_box = max(1.0, max(ys) - min(ys))
            area = w_box * h_box
            cx = (x_min + x_max)/2.0
        rel_area = area / (w_img * h_img + 1e-9)
        dist_norm = abs(cx - center_x_img) / (w_img + 1e-9)
        nl = len(num_sub)
        score = nl * 10.0 + rel_area * 1000.0 - dist_norm * 20.0 + float(prob) * 2.0
        candidates.append({
            "raw_text": raw_text,
            "cleaned": cleaned,
            "num_sub": num_sub,
            "num_len": nl,
            "prob": prob,
            "area": area,
            "rel_area": rel_area,
            "cx": cx,
            "dist_norm": dist_norm,
            "score": score,
            "bbox": bbox,
        })

    if not candidates:
        return "UNKNOWN", "TIDAK TERDEFINISI"

    candidates.sort(key=lambda x: x["score"], reverse=True)

    selected = None
    for cand in candidates:
        if 2 <= cand["num_len"] <= 4:
            selected = cand
            break
    if selected is None:
        selected = candidates[0]

    main_number = selected["num_sub"]

    color_key, vehicle_label = detect_plate_color(plate_img)
    return main_number, vehicle_label

# ---------------- UI / Layout (mockup-like) ----------------
root = tk.Tk()
root.title("ALPR_TPS - Gate IN System")
root.geometry("1280x720")

# Grid config to mimic mockup (left column narrow, right big)
root.grid_columnconfigure(0, weight=1, minsize=340)
root.grid_columnconfigure(1, weight=3)
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=0, minsize=80)

# Left column panel (stacked containers)
panel_left = ttk.Frame(root, padding=8, relief="flat")
panel_left.grid(row=0, column=0, sticky="nsew")

# Container: captured image preview (top-left)
frame_preview = ttk.LabelFrame(panel_left, text="SHOW IMAGE\nCAPTURED IMAGES WITHOUT BBOX", padding=8)
frame_preview.pack(fill="both", expand=False, pady=(0,8))

preview_label = tk.Label(frame_preview, text="No image", width=40, height=30, bg="#f2f2f2")
preview_label.pack(fill="both", expand=True)

# Container: detected character & type (mid-left)
frame_char = ttk.LabelFrame(panel_left, text="SHOW CHARACTER\nDETECTED LICENSE PLATE NUMBER AND VEHICLE TYPE", padding=8)
frame_char.pack(fill="both", expand=False, pady=(0,8))

lbl_plate_text = tk.Label(frame_char, text="Plate: -", font=("Helvetica", 14))
lbl_plate_text.pack(pady=(8,4))

lbl_vehicle_type = tk.Label(frame_char, text="Type: -", font=("Helvetica", 12))
lbl_vehicle_type.pack(pady=(0,8))

# Container: datetime (lower-left)
frame_dt = ttk.LabelFrame(panel_left, text="SHOW DATETIME\nVEHICLE CAPTURED", padding=8)
frame_dt.pack(fill="both", expand=False, pady=(0,8))

lbl_datetime = tk.Label(frame_dt, text="-", font=("Helvetica", 11))
lbl_datetime.pack(pady=(8,8))

# Right: large video panel
frame_video = ttk.LabelFrame(root, text="SHOW VIDEO SOURCE\nLive tracking with bounding box", padding=4)
frame_video.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
frame_video.grid_rowconfigure(0, weight=1)
frame_video.grid_columnconfigure(0, weight=1)

video_label = tk.Label(frame_video, bg="black")
video_label.grid(row=0, column=0, sticky="nsew")

# Bottom: instructions/status bar
frame_bottom = ttk.Frame(root, padding=6)
frame_bottom.grid(row=1, column=0, columnspan=2, sticky="ew")
lbl_instructions = tk.Label(frame_bottom,
    text="SHOW TEXT  |  Instructions: Press C to Capture | Press Q to Close App",
    font=("Helvetica", 10))
lbl_instructions.pack(fill="x")

# ---------------- Video capture ----------------
cap = cv2.VideoCapture(VIDEO_SOURCE)
# If VIDEO_SOURCE is integer path string like "0", convert to int
try:
    if isinstance(VIDEO_SOURCE, str) and VIDEO_SOURCE.isdigit():
        cap = cv2.VideoCapture(int(VIDEO_SOURCE))
except Exception:
    pass

last_plate_crop = None
last_frame_for_save = None
last_selected_bbox = None  # store bbox of selected candidate for visualization if needed

# Helper: preserve aspect ratio fit into box (target_w,target_h)
def resize_keep_aspect(img, target_w, target_h):
    h, w = img.shape[:2]
    if w == 0 or h == 0:
        return img
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # create canvas with target size and center the image
    canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas

# ---------------- Main loop update ----------------
def update_frame():
    global last_plate_crop, last_frame_for_save, last_selected_bbox

    ret, frame = cap.read()
    if not ret:
        # try again later
        root.after(50, update_frame)
        return

    # store latest full frame for saving when capture pressed
    last_frame_for_save = frame.copy()

    # Run YOLO detection (use stream=False to get full result)
    try:
        results = yolo_model(frame, stream=False)
    except Exception as e:
        # if model fails, still show frame
        results = None
        print("YOLO infer error:", e)

    # reset last_plate_crop (if none found) to keep UI behavior consistent
    found_plate_this_frame = False
    if results is not None and len(results) > 0:
        # results[0].boxes might be empty if no detection
        boxes = getattr(results[0], "boxes", None)
        if boxes is not None and len(boxes) > 0:
            # pick the biggest box (area) to be main plate crop for preview
            best_box = None
            best_area = -1
            for b in boxes:
                try:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                except Exception:
                    continue
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best_box = (x1, y1, x2, y2)
            if best_box is not None:
                x1, y1, x2, y2 = best_box
                # keep crop for capture, without bbox overlay saved
                last_plate_crop = frame[y1:y2, x1:x2].copy() if (y2>y1 and x2>x1) else None
                last_selected_bbox = best_box
                found_plate_this_frame = True
                # draw bbox on display frame (not on saved crop)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display frame into video_label, preserving aspect ratio of source inside the widget area
    target_w = max(1, video_label.winfo_width())
    target_h = max(1, video_label.winfo_height())
    # If widget not yet realized, assign defaults
    if target_w < 10 or target_h < 10:
        target_w, target_h = 960, 540  # default area

    frame_to_show = resize_keep_aspect(frame, target_w, target_h)
    img_rgb = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    imgtk = ImageTk.PhotoImage(image=img_pil)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Update preview thumbnail (left-top) keep aspect ratio and small size
    if last_plate_crop is not None:
        thumb_w, thumb_h = 300, 100
        thumb = resize_keep_aspect(last_plate_crop, thumb_w, thumb_h)
        thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
        thumb_pil = Image.fromarray(thumb_rgb)
        thumb_tk = ImageTk.PhotoImage(image=thumb_pil)
        preview_label.config(image=thumb_tk, text="")
        preview_label.image = thumb_tk
    else:
        preview_label.config(text="No plate", image="", bg="#f2f2f2")

    # schedule next
    root.after(30, update_frame)

# ---------------- Capture action (C) ----------------
def capture_and_log():
    global last_plate_crop, last_frame_for_save, last_selected_bbox
    if last_plate_crop is None:
        lbl_instructions.config(text="No plate detected. Aim camera at plate and press C.")
        return

    # run OCR + selection
    plate_number, vehicle_type = process_ocr_and_pick_number(last_frame_for_save, last_plate_crop)
    if plate_number is None:
        plate_number = "UNKNOWN"
    if vehicle_type is None:
        vehicle_type = "TIDAK TERDEFINISI"

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename = f"capture_{timestamp}.jpg"
    save_path = os.path.join(CAPTURE_DIR, filename)

    # Save the full frame WITHOUT drawn bbox (we used last_frame_for_save which is original)
    if last_frame_for_save is not None:
        try:
            cv2.imwrite(save_path, last_frame_for_save)
            lbl_instructions.config(text=f"Saved capture: {save_path}")
        except Exception as e:
            lbl_instructions.config(text=f"Failed save: {e}")
    else:
        lbl_instructions.config(text="No frame available to save.")

    # Append log
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([date_str, time_str, plate_number, vehicle_type, filename])

    # Update left UI: plate text, vehicle type, datetime & keep preview
    lbl_plate_text.config(text=f"Plate: {plate_number}")
    lbl_vehicle_type.config(text=f"Type: {vehicle_type}")
    lbl_datetime.config(text=f"{date_str} {time_str}")

    # Optionally draw bbox on saved preview (visual in a separate window) -- we will overlay number on thumbnail in UI:
    # create thumbnail with large bounding box and number overlay to mimic cell5 visualization
    if last_plate_crop is not None:
        vis = last_plate_crop.copy()
        # draw bbox on preview (full crop)
        if last_selected_bbox is not None:
            # try to draw selected bbox text near top-left corner
            cv2.putText(vis, plate_number, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        # resize for preview
        thumb = resize_keep_aspect(vis, 300, 100)
        thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
        thumb_pil = Image.fromarray(thumb_rgb)
        thumb_tk = ImageTk.PhotoImage(image=thumb_pil)
        preview_label.config(image=thumb_tk, text="")
        preview_label.image = thumb_tk

# ---------------- Key bindings + Buttons ----------------
def on_key(event):
    key = event.char.lower()
    if key == "q":
        try:
            cap.release()
        except:
            pass
        root.quit()
        root.destroy()
    elif key == "c":
        capture_and_log()

root.bind("<Key>", on_key)

# Also provide a capture button for convenience
btn_frame = ttk.Frame(frame_bottom)
btn_frame.pack(pady=4)
btn_capture = ttk.Button(btn_frame, text="Capture (C)", command=capture_and_log)
btn_quit = ttk.Button(btn_frame, text="Quit (Q)", command=lambda: on_key(type("Evt",(object,),{"char":"q"})()))
btn_capture.pack(side="left", padx=8)
btn_quit.pack(side="left", padx=8)

# ---------------- Start ----------------
update_frame()
root.mainloop()

# Cleanup on exit
try:
    cap.release()
    cv2.destroyAllWindows()
except:
    pass
