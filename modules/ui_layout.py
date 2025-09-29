import cv2
import numpy as np
from modules.utils_io import resize_keep_aspect


def create_dashboard(live_frame, capture_data, plates_info):
    """
    Buat dashboard UI lengkap dengan layout responsive
    """
    # Dapatkan ukuran window saat ini
    try:
        window_size = cv2.getWindowImageRect("ALPR System - PT TPS Surabaya")
        if window_size[2] > 0 and window_size[3] > 0:
            dashboard_width = max(1000, window_size[2])  # Minimum width 1000
            dashboard_height = max(600, window_size[3])  # Minimum height 600
        else:
            # Fallback ke ukuran default
            dashboard_width = 1200
            dashboard_height = 700
    except:
        # Jika window belum ada atau error, gunakan ukuran default
        dashboard_width = 1200
        dashboard_height = 700

    # Buat canvas dashboard
    dashboard = (
        np.ones((dashboard_height, dashboard_width, 3), dtype=np.uint8) * 240
    )  # Light gray background

    # === BAGIAN KIRI: INFO PANEL ===
    panel_width = min(400, int(dashboard_width * 0.33))  # Maksimal 33% lebar window
    panel_height = dashboard_height

    # Buat panel kiri dengan background putih
    left_panel = np.ones((panel_height, panel_width, 3), dtype=np.uint8) * 255

    # --- KOTAK 1: SHOW IMAGE (Captured Plate) ---
    box1_height = int(panel_height * 0.25)  # 25% dari tinggi panel
    cv2.rectangle(
        left_panel, (10, 10), (panel_width - 10, box1_height - 10), (200, 200, 200), 2
    )
    cv2.putText(
        left_panel,
        "CAPTURED PLATE IMAGE",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
    )

    # Tampilkan gambar plat yang dicapture (jika ada)
    if capture_data["plate_image"] is not None:
        plate_display = resize_keep_aspect(
            capture_data["plate_image"], panel_width - 40, box1_height - 50
        )
        ph, pw = plate_display.shape[:2]
        y_offset = 40
        x_offset = (panel_width - pw) // 2
        # Pastikan tidak melebihi batas
        if y_offset + ph <= box1_height - 10 and x_offset + pw <= panel_width - 10:
            left_panel[y_offset : y_offset + ph, x_offset : x_offset + pw] = (
                plate_display
            )
    else:
        # Tampilkan placeholder jika belum ada capture
        placeholder_text = "No Capture Yet"
        text_size = cv2.getTextSize(placeholder_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[
            0
        ]
        text_x = (panel_width - text_size[0]) // 2
        text_y = (box1_height - 10 + 30) // 2
        cv2.putText(
            left_panel,
            placeholder_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (150, 150, 150),
            1,
        )

    # --- KOTAK 2: SHOW CHARACTER & VEHICLE TYPE ---
    box2_top = box1_height + 10
    box2_height = int(panel_height * 0.25)
    cv2.rectangle(
        left_panel,
        (10, box2_top),
        (panel_width - 10, box2_top + box2_height - 10),
        (200, 200, 200),
        2,
    )
    cv2.putText(
        left_panel,
        "DETECTED LICENSE PLATE",
        (20, box2_top + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
    )

    # Tampilkan plat number dengan confidence
    plate_text = capture_data["plate_number"]
    vehicle_text = capture_data["vehicle_type"]
    confidence = capture_data.get("confidence", 0.0)

    # Adjust font size berdasarkan panjang teks
    plate_font_size = 0.6 if len(plate_text) < 15 else 0.4
    vehicle_font_size = 0.5 if len(vehicle_text) < 30 else 0.35

    # Tampilkan plat number
    plate_display_text = f"PLATE: {plate_text}"
    cv2.putText(
        left_panel,
        plate_display_text,
        (30, box2_top + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        plate_font_size,
        (0, 100, 0),
        2,
    )

    # Tampilkan confidence score
    confidence_text = f"Confidence: {confidence:.2f}" if confidence > 0 else ""
    cv2.putText(
        left_panel,
        confidence_text,
        (30, box2_top + 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 0, 200),
        1,
    )

    # Tampilkan vehicle type
    cv2.putText(
        left_panel,
        f"TYPE: {vehicle_text}",
        (30, box2_top + 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        vehicle_font_size,
        (0, 0, 200),
        1,
    )

    # --- KOTAK 3: SHOW DATETIME ---
    box3_top = box2_top + box2_height + 10
    box3_height = int(panel_height * 0.15)
    cv2.rectangle(
        left_panel,
        (10, box3_top),
        (panel_width - 10, box3_top + box3_height - 10),
        (200, 200, 200),
        2,
    )
    cv2.putText(
        left_panel,
        "CAPTURE TIMESTAMP",
        (20, box3_top + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
    )

    time_text = (
        capture_data["capture_time"]
        if capture_data["capture_time"]
        else "BELUM ADA CAPTURE"
    )
    time_font_size = 0.5 if len(time_text) < 25 else 0.35
    cv2.putText(
        left_panel,
        time_text,
        (30, box3_top + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        time_font_size,
        (100, 0, 100),
        1,
    )

    # --- STATISTICS ---
    box4_top = box3_top + box3_height + 10
    box4_height = int(panel_height * 0.15)
    cv2.rectangle(
        left_panel,
        (10, box4_top),
        (panel_width - 10, box4_top + box4_height - 10),
        (200, 200, 200),
        2,
    )
    cv2.putText(
        left_panel,
        "SYSTEM STATUS",
        (20, box4_top + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
    )

    plates_detected = len(plates_info)
    status_color = (0, 150, 0) if plates_detected > 0 else (0, 0, 150)
    status_text = f"PLATES DETECTED: {plates_detected}"

    cv2.putText(
        left_panel,
        status_text,
        (30, box4_top + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        status_color,
        1,
    )

    # --- CAPTURE COUNTER ---
    box5_top = box4_top + box4_height + 10
    box5_height = int(panel_height * 0.1)
    cv2.rectangle(
        left_panel,
        (10, box5_top),
        (panel_width - 10, box5_top + box5_height - 10),
        (200, 200, 200),
        2,
    )
    cv2.putText(
        left_panel,
        "CONTROLS",
        (20, box5_top + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
    )

    controls_text = "Press C to CAPTURE"
    cv2.putText(
        left_panel,
        controls_text,
        (30, box5_top + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 100, 0),
        1,
    )

    # === BAGIAN KANAN: LIVE VIDEO ===
    right_width = dashboard_width - panel_width
    right_panel = np.ones((dashboard_height, right_width, 3), dtype=np.uint8) * 255

    # --- LIVE VIDEO FEED ---
    video_height = int(dashboard_height * 0.7)  # 70% dari tinggi window
    cv2.rectangle(
        right_panel, (10, 10), (right_width - 10, video_height), (100, 100, 100), 2
    )
    cv2.putText(
        right_panel,
        "LIVE VIDEO FEED - PLATE DETECTION",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
    )

    # Tampilkan live frame
    if live_frame is not None:
        live_display = resize_keep_aspect(
            live_frame, right_width - 40, video_height - 40
        )
        lh, lw = live_display.shape[:2]
        y_offset = 40
        x_offset = (right_width - lw) // 2
        # Pastikan tidak melebihi batas
        if y_offset + lh <= video_height and x_offset + lw <= right_width - 10:
            right_panel[y_offset : y_offset + lh, x_offset : x_offset + lw] = (
                live_display
            )

    # --- DETECTION INFO ---
    info_top = video_height + 10
    info_height = int(dashboard_height * 0.15)
    cv2.rectangle(
        right_panel,
        (10, info_top),
        (right_width - 10, info_top + info_height),
        (180, 180, 180),
        2,
    )
    cv2.putText(
        right_panel,
        "DETECTION INFORMATION",
        (20, info_top + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
    )

    # Tampilkan info deteksi
    detection_info = []
    if plates_detected > 0:
        best_plate = max(plates_info, key=lambda x: x["confidence"])
        detection_info = [
            f"Best Detection: Confidence {best_plate['confidence']:.2f}",
            f"Total Detections: {plates_detected}",
            "Status: READY FOR CAPTURE (Press C)",
        ]
    else:
        detection_info = [
            "No plates detected",
            "Position vehicle clearly",
            "Status: WAITING FOR PLATES",
        ]

    y_pos = info_top + 40
    for info_line in detection_info:
        cv2.putText(
            right_panel,
            info_line,
            (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
        )
        y_pos += 20

    # --- INSTRUCTIONS ---
    instructions_top = info_top + info_height + 10
    instructions_height = dashboard_height - instructions_top - 10
    cv2.rectangle(
        right_panel,
        (10, instructions_top),
        (right_width - 10, instructions_top + instructions_height),
        (150, 150, 150),
        2,
    )
    cv2.putText(
        right_panel,
        "ALPR SYSTEM - PT TPS SURABAYA",
        (20, instructions_top + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 150),
        1,
    )

    instructions = [
        "INSTRUCTIONS:",
        "1. Press 'C' key to CAPTURE detected plate",
        "2. Press 'ESC' key to CLOSE application",
        "3. Close with [X] button on window",
        "4. Resize window for better view",
        "",
        "Features:",
        "- Automatic license plate detection",
        "- Advanced OCR character recognition",
        "- Vehicle type classification by plate color",
        "- Real-time timestamp logging",
    ]

    y_pos = instructions_top + 50
    line_height = 18  # Sedikit lebih rapat

    for line in instructions:
        # Adjust font size untuk instruksi
        font_size = 0.35
        cv2.putText(
            right_panel,
            line,
            (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (0, 0, 0),
            1,
        )
        y_pos += line_height

    # Gabungkan panel kiri dan kanan
    dashboard[0:panel_height, 0:panel_width] = left_panel
    dashboard[0:dashboard_height, panel_width:dashboard_width] = right_panel

    return dashboard


def close_windows():
    """
    Tutup semua jendela OpenCV.
    """
    cv2.destroyAllWindows()
