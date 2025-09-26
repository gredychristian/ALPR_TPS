import cv2
import re
import os
import csv
import tkinter as tk
from tkinter import Label, Frame
from PIL import Image, ImageTk
from ultralytics import YOLO
from datetime import datetime
import easyocr
import numpy as np

# ---------------- CONFIG ----------------
VIDEO_SOURCE = 0
MODEL_PATH = "runs/detect/train/weights/best.pt"
CAPTURE_DIR = "capture_gatein"
LOG_FILE = "log_gatein.csv"

if not os.path.exists(CAPTURE_DIR):
    os.makedirs(CAPTURE_DIR)

# Inisialisasi log CSV bila belum ada
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "time", "plate_number", "vehicle_type", "filename"])

# Load YOLO model
yolo_model = YOLO(MODEL_PATH)

# Load EasyOCR
reader = easyocr.Reader(['en'])

# ---------------- HELPER FUNCS ----------------
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

def detect_plate_color(img):
    """Dummy fungsi untuk deteksi warna → sesuaikan kebutuhanmu"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    avg_v = hsv[...,2].mean()
    if avg_v > 128:
        return "white", "mobil"
    else:
        return "black", "motor"

def process_ocr(img):
    results = reader.readtext(img)
    h_img, w_img = img.shape[:2]
    center_x_img = w_img / 2.0

    candidates = []
    for bbox, text, prob in results:
        raw_text = str(text)
        cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_text).upper()
        nums = numeric_substrings(cleaned)
        if len(nums) == 0:
            continue
        num_sub = max(nums, key=len)
        x_min, y_min, x_max, y_max, w_box, h_box, area, cx, cy = bbox_metrics(bbox)
        rel_area = area / (w_img * h_img)
        dist_norm = abs(cx - center_x_img) / w_img
        nl = len(num_sub)
        score = nl * 10.0 + rel_area * 1000.0 - dist_norm * 20.0 + float(prob) * 2.0
        candidates.append({
            "raw_text": raw_text,
            "num_sub": num_sub,
            "num_len": nl,
            "prob": float(prob),
            "score": score,
            "bbox": bbox,
        })

    if not candidates:
        return "UNKNOWN", "UNKNOWN"

    candidates.sort(key=lambda c: c["score"], reverse=True)

    selected = None
    for cand in candidates:
        if 2 <= cand["num_len"] <= 4:
            selected = cand
            break
    if selected is None:
        selected = candidates[0]

    main_number = selected["num_sub"]

    color, jenis = detect_plate_color(img)
    return main_number, jenis

# ---------------- GUI SETUP ----------------
root = tk.Tk()
root.title("ALPR_TPS - Gate IN System")
root.geometry("1280x720")

# layout grid
root.grid_columnconfigure(0, weight=1, minsize=320)
root.grid_columnconfigure(1, weight=3)
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=0)

# Frame kiri (info plat)
frame_left = Frame(root, bd=2, relief="solid")
frame_left.grid(row=0, column=0, sticky="nsew")

lbl_show_image = Label(frame_left, text="SHOW IMAGE\nPlate crop here", font=("Arial", 10))
lbl_show_image.pack(fill="x", pady=5)

lbl_show_character = Label(frame_left, text="SHOW CHARACTER\nPlate number", font=("Arial", 12))
lbl_show_character.pack(fill="x", pady=5)

lbl_datetime = Label(frame_left, text="SHOW DATETIME\nVehicle Capture", font=("Arial", 10))
lbl_datetime.pack(fill="x", pady=5)

# Frame kanan (video)
frame_right = Frame(root, bd=2, relief="solid")
frame_right.grid(row=0, column=1, sticky="nsew")

video_label = Label(frame_right)
video_label.pack(fill="both", expand=True)

# Frame bawah (info)
frame_bottom = Frame(root, bd=2, relief="solid")
frame_bottom.grid(row=1, column=0, columnspan=2, sticky="ew")

lbl_text = Label(frame_bottom,
    text="SHOW TEXT\nInstructions: Press C to Capture | Press Q to Close App",
    font=("Arial", 10))
lbl_text.pack(fill="x")

# ---------------- VIDEO LOOP ----------------
cap = cv2.VideoCapture(VIDEO_SOURCE)
last_plate_crop = None

def update_frame():
    global last_plate_crop
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    results = yolo_model(frame, stream=False)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        last_plate_crop = frame[y1:y2, x1:x2].copy()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # === FIX aspect ratio untuk video ===
    h_frame, w_frame = frame.shape[:2]
    w_label = video_label.winfo_width()
    h_label = video_label.winfo_height()

    # fallback kalau belum dirender
    if w_label <= 1 or h_label <= 1:
        w_label, h_label = 640, 480  

    scale = min(w_label / w_frame, h_label / h_frame)
    new_w, new_h = max(1, int(w_frame * scale)), max(1, int(h_frame * scale))

    frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h_label, w_label, 3), dtype=np.uint8)
    y_offset = (h_label - new_h) // 2
    x_offset = (w_label - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = frame_resized

    img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = img_tk
    video_label.configure(image=img_tk)

    root.after(10, update_frame)


def capture_and_process():
    global last_plate_crop
    if last_plate_crop is None:
        lbl_text.config(text="No plate detected to capture.")
        return

    plate_number, vehicle_type = process_ocr(last_plate_crop)
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # simpan foto
    ret, frame = cap.read()
    filename = f"capture_{timestamp}.jpg"
    save_path = os.path.join(CAPTURE_DIR, filename)
    if ret:
        cv2.imwrite(save_path, frame)
        lbl_text.config(text=f"Saved capture: {save_path}")

    # === FIX aspect ratio untuk preview crop ===
    img_crop = cv2.cvtColor(last_plate_crop, cv2.COLOR_BGR2RGB)
    h_crop, w_crop = img_crop.shape[:2]
    target_w, target_h = 300, 100
    scale = min(target_w / w_crop, target_h / h_crop)
    new_w, new_h = int(w_crop * scale), int(h_crop * scale)

    resized_crop = cv2.resize(img_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_crop

    img_crop_pil = Image.fromarray(canvas)
    img_crop_tk = ImageTk.PhotoImage(image=img_crop_pil)
    lbl_show_image.config(image=img_crop_tk, text="")
    lbl_show_image.image = img_crop_tk

    lbl_show_character.config(text=f"Plate: {plate_number}")
    lbl_datetime.config(text=f"Captured: {date_str} {time_str}")

    # tulis log
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([date_str, time_str, plate_number, vehicle_type, filename])


# ---------------- KEY BIND ----------------
def on_key(event):
    if event.char.lower() == "q":
        cap.release()
        root.destroy()
    elif event.char.lower() == "c":
        capture_and_process()

root.bind("<Key>", on_key)

# start loop
update_frame()
root.mainloop()
