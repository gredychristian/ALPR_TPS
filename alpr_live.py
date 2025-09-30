# alpr_live.py
import cv2
import os
import sys
import numpy as np
from datetime import datetime
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

from utils.plate_detector import PlateDetector
from utils.ocr_reader import OCRReader
import config


class ALPRLiveSystem:
    def __init__(self):
        print("🚀 Initializing ALPR Live System...")
        self.plate_detector = PlateDetector(config.PLATE_DETECTOR_MODEL)
        self.ocr_reader = OCRReader()
        print("✅ ALPR Live System Ready!")

        # State variables
        self.captured_plate_image = None
        self.capture_time = None
        self.ocr_text = ""
        self.ocr_confidence = 0.0
        self.last_detection = None

        # ✅ CSV LOGGING
        self.log_file = os.path.join(config.OUTPUT_DIR, "log.csv")
        self.initialize_log_file()

        # Flexible window configuration
        self.min_width = 800
        self.min_height = 600
        self.aspect_ratio = 4.0 / 3.0
        self.current_width = self.min_width
        self.current_height = self.min_height
        self.side_panel_ratio = 0.33

    def initialize_log_file(self):
        """
        Initialize CSV log file dengan headers jika belum ada
        """
        if not os.path.exists(self.log_file):
            log_df = pd.DataFrame(
                columns=["timestamp", "license_plate", "confidence", "filename"]
            )
            log_df.to_csv(self.log_file, index=False)
            print(f"📝 Created new log file: {self.log_file}")
        else:
            print(f"📝 Using existing log file: {self.log_file}")

    def log_capture(self, license_plate, confidence, filename):
        """
        Log capture data ke CSV file
        """
        try:
            # Baca existing log
            log_df = pd.read_csv(self.log_file)

            # Tambah entry baru
            new_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "license_plate": license_plate,
                "confidence": confidence,
                "filename": filename,
            }

            # Append ke DataFrame
            log_df = pd.concat([log_df, pd.DataFrame([new_entry])], ignore_index=True)

            # Save ke CSV
            log_df.to_csv(self.log_file, index=False)

            print(f"📝 Logged to CSV: {license_plate} (Conf: {confidence:.2f})")

        except Exception as e:
            print(f"❌ Error logging to CSV: {e}")

    def calculate_layout(self):
        """
        Calculate dynamic layout berdasarkan current window size
        """
        # Hitung side panel width berdasarkan ratio
        side_panel_width = int(self.current_width * self.side_panel_ratio)
        main_width = self.current_width - side_panel_width

        # Pastikan minimum size
        side_panel_width = max(300, side_panel_width)
        main_width = max(500, main_width)

        return {
            "main_width": main_width,
            "main_height": self.current_height,
            "side_panel_width": side_panel_width,
            "panel_height": self.current_height // 2,
        }

    def resize_with_aspect_ratio(
        self, image, width=None, height=None, inter=cv2.INTER_AREA
    ):
        """
        Resize image dengan maintain aspect ratio
        """
        dim = None
        h, w = image.shape[:2]

        if width is None and height is None:
            return image, 1.0

        if width is None:
            # Calculate ratio based on height
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            # Calculate ratio based on width
            r = width / float(w)
            dim = (width, int(h * r))

        resized = cv2.resize(image, dim, interpolation=inter)
        return resized, r

    def create_dashboard(self, main_frame, plates):
        """
        Create 3-panel dashboard dengan dynamic layout
        """
        # Dapatkan layout berdasarkan current window size
        layout = self.calculate_layout()
        main_width = layout["main_width"]
        main_height = layout["main_height"]
        side_panel_width = layout["side_panel_width"]
        panel_height = layout["panel_height"]

        # Create blank canvas
        dashboard_height = main_height
        dashboard_width = main_width + side_panel_width
        dashboard = (
            np.ones((dashboard_height, dashboard_width, 3), dtype=np.uint8) * 240
        )

        # ===== PANEL 1: LIVE VIDEO SOURCE (Kiri) =====
        panel1_x, panel1_y = 0, 0
        panel1_width, panel1_height = main_width, main_height

        # Resize dengan maintain aspect ratio video source
        main_resized, scale = self.resize_with_aspect_ratio(
            main_frame, width=panel1_width, height=panel1_height
        )

        # Calculate padding untuk center the image
        h, w = main_resized.shape[:2]
        pad_x = (panel1_width - w) // 2
        pad_y = (panel1_height - h) // 2

        # Create panel dengan padding
        panel1_bg = np.ones((panel1_height, panel1_width, 3), dtype=np.uint8) * 50

        # Pastikan tidak melebihi boundary
        end_y = min(pad_y + h, panel1_height)
        end_x = min(pad_x + w, panel1_width)
        actual_h = end_y - pad_y
        actual_w = end_x - pad_x

        if actual_h > 0 and actual_w > 0:
            panel1_bg[pad_y:end_y, pad_x:end_x] = main_resized[:actual_h, :actual_w]

        # Draw bounding boxes jika ada plates
        if plates:
            for plate in plates:
                x1, y1, x2, y2 = plate["bbox"]
                conf = plate["confidence"]

                # Adjust coordinates berdasarkan scaling dan padding
                x1_adj = int(x1 * scale) + pad_x
                y1_adj = int(y1 * scale) + pad_y
                x2_adj = int(x2 * scale) + pad_x
                y2_adj = int(y2 * scale) + pad_y

                # Pastikan coordinates dalam boundary dan valid
                x1_adj = max(0, min(x1_adj, panel1_width - 1))
                y1_adj = max(0, min(y1_adj, panel1_height - 1))
                x2_adj = max(0, min(x2_adj, panel1_width - 1))
                y2_adj = max(0, min(y2_adj, panel1_height - 1))

                # ✅ PERBAIKAN: Pastikan coordinates valid untuk drawing
                if (
                    x2_adj > x1_adj
                    and y2_adj > y1_adj
                    and (x2_adj - x1_adj) > 5
                    and (y2_adj - y1_adj) > 5
                ):
                    try:
                        # 🎨 COLOR SETTINGS
                        color = (0, 210, 0)  # 🔵 BIRU TERANG (Tosca)
                        box_thickness = 3  # 📏 Medium thickness
                        font_scale = 0.5  # 🔤 Normal font size
                        text_thickness = 2  # 📝 Medium text thickness
                        bg_color = (0, 210, 0)  # 🔳 Black background
                        text_color = (255, 255, 255)  # ⚪ White text

                        # 1. Draw main bounding box
                        cv2.rectangle(
                            panel1_bg,
                            (x1_adj, y1_adj),
                            (x2_adj, y2_adj),
                            color,
                            box_thickness,
                        )

                        # 2. Prepare label text
                        label_text = f"License Plate {conf:.2f}"

                        # 3. Calculate text size
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label_text,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            text_thickness,
                        )

                        # 4. Calculate background rectangle coordinates
                        bg_x1 = x1_adj
                        bg_y1 = max(0, y1_adj - text_height - 10)  # 10px margin atas
                        bg_x2 = x1_adj + text_width + 10  # 10px padding kanan-kiri
                        bg_y2 = y1_adj

                        # ✅ PERBAIKAN: Pastikan background tidak keluar dari frame
                        if bg_y1 < 0:
                            bg_y1 = y1_adj
                            bg_y2 = y1_adj + text_height + 10

                        # 5. Draw filled black background
                        cv2.rectangle(
                            panel1_bg,
                            (bg_x1, bg_y1),
                            (bg_x2, bg_y2),
                            bg_color,
                            -1,  # Filled rectangle
                        )

                        # 6. Calculate text position
                        text_x = bg_x1 + 5  # 5px padding dari kiri background
                        text_y = bg_y2 - 5  # 5px dari bawah background

                        # ✅ PERBAIKAN: Jika background di atas, adjust text position
                        if bg_y1 < y1_adj:
                            text_y = bg_y2 - 5
                        else:
                            text_y = bg_y1 + text_height + 5

                        # 7. Draw white text
                        cv2.putText(
                            panel1_bg,
                            label_text,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            text_color,
                            text_thickness,
                        )

                    except Exception as e:
                        print(f"⚠️  Error drawing bounding box: {e}")
                        # Fallback: simple rectangle tanpa label
                        cv2.rectangle(
                            panel1_bg,
                            (x1_adj, y1_adj),
                            (x2_adj, y2_adj),
                            (0, 255, 255),
                            2,
                        )

        # Add panel title
        title_scale = min(0.8, 0.6 * (main_width / 800))
        cv2.putText(
            panel1_bg,
            "LIVE TRACKING WITH BOUNDING BOX",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            title_scale,
            (255, 255, 255),
            2,
        )

        # Place panel 1
        dashboard[
            panel1_y : panel1_y + panel1_height, panel1_x : panel1_x + panel1_width
        ] = panel1_bg

        # ===== PANEL 2: CAPTURED PLATE (Kanan Atas) =====
        panel2_x, panel2_y = main_width, 0
        panel2_width, panel2_height = side_panel_width, panel_height

        # Create panel 2
        panel2 = np.ones((panel2_height, panel2_width, 3), dtype=np.uint8) * 255

        # Add title
        cv2.putText(
            panel2,
            "CAPTURED PLATE IMAGE",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            title_scale,
            (0, 0, 0),
            2,
        )

        # Display captured plate image jika ada
        if self.captured_plate_image is not None:
            try:
                plate_h, plate_w = self.captured_plate_image.shape[:2]

                # Calculate maximum dimensions dengan margin
                max_width = panel2_width - 40
                max_height = panel2_height - 60

                # Resize dengan maintain aspect ratio
                plate_resized, plate_scale = self.resize_with_aspect_ratio(
                    self.captured_plate_image, width=max_width, height=max_height
                )

                # Center the image in panel
                res_h, res_w = plate_resized.shape[:2]
                start_x = (panel2_width - res_w) // 2
                start_y = (panel2_height - res_h) // 2 + 20

                # Pastikan tidak melebihi boundary panel
                end_y = min(start_y + res_h, panel2_height)
                end_x = min(start_x + res_w, panel2_width)
                actual_h = end_y - start_y
                actual_w = end_x - start_x

                if actual_h > 0 and actual_w > 0:
                    panel2[start_y:end_y, start_x:end_x] = plate_resized[
                        :actual_h, :actual_w
                    ]

                    # Add border
                    cv2.rectangle(
                        panel2,
                        (start_x - 2, start_y - 2),
                        (
                            min(start_x + res_w + 2, panel2_width - 1),
                            min(start_y + res_h + 2, panel2_height - 1),
                        ),
                        (0, 0, 0),
                        2,
                    )

            except Exception as e:
                print(f"⚠️  Error displaying captured plate: {e}")
                cv2.putText(
                    panel2,
                    "Error displaying image",
                    (panel2_width // 2 - 100, panel2_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (100, 100, 100),
                    1,
                )
        else:
            # Placeholder text
            cv2.putText(
                panel2,
                "No plate captured",
                (panel2_width // 2 - 80, panel2_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (100, 100, 100),
                1,
            )
            cv2.putText(
                panel2,
                "Press 'C' to capture",
                (panel2_width // 2 - 90, panel2_height // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (100, 100, 100),
                1,
            )

        # Place panel 2
        dashboard[
            panel2_y : panel2_y + panel2_height, panel2_x : panel2_x + panel2_width
        ] = panel2

        # ===== PANEL 3: OCR RESULT (Kanan Bawah) =====
        panel3_x, panel3_y = main_width, panel_height
        panel3_width, panel3_height = side_panel_width, panel_height

        # Create panel 3
        panel3 = np.ones((panel3_height, panel3_width, 3), dtype=np.uint8) * 255

        # Add title
        cv2.putText(
            panel3,
            "DETECTED LICENSE PLATE",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            title_scale,
            (0, 0, 0),
            2,
        )

        # Display OCR results jika ada
        if self.ocr_text:
            # Plate number (scale font berdasarkan window size)
            font_scale = min(2.0, 1.5 * (side_panel_width / 400))
            text_size = cv2.getTextSize(
                self.ocr_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3
            )[0]
            text_x = (panel3_width - text_size[0]) // 2
            text_y = panel3_height // 2 - 20

            cv2.putText(
                panel3,
                self.ocr_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 100, 0),
                3,
            )

            # Confidence score
            conf_text = f"Confidence: {self.ocr_confidence:.2f}"
            cv2.putText(
                panel3,
                conf_text,
                (panel3_width // 2 - 80, text_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
            )

            # Timestamp
            if self.capture_time:
                time_text = f"Captured: {self.capture_time}"
                cv2.putText(
                    panel3,
                    time_text,
                    (20, panel3_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (100, 100, 100),
                    1,
                )
        else:
            # Placeholder text
            cv2.putText(
                panel3,
                "No OCR result",
                (panel3_width // 2 - 60, panel3_height // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (100, 100, 100),
                1,
            )
            cv2.putText(
                panel3,
                "Press 'C' to capture & recognize",
                (panel3_width // 2 - 120, panel3_height // 2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (100, 100, 100),
                1,
            )

        # Place panel 3
        dashboard[
            panel3_y : panel3_y + panel3_height, panel3_x : panel3_x + panel3_width
        ] = panel3

        # ===== DIVIDER LINES =====
        cv2.line(
            dashboard, (main_width, 0), (main_width, main_height), (200, 200, 200), 2
        )
        cv2.line(
            dashboard,
            (main_width, panel_height),
            (main_width + side_panel_width, panel_height),
            (200, 200, 200),
            2,
        )

        return dashboard

    def capture_and_recognize(self, plate, original_frame):
        """
        Capture plate dan lakukan OCR recognition dengan logging
        """
        try:
            print("📸 Capturing plate for OCR...")

            # Capture plate image dengan boundary check
            x1, y1, x2, y2 = plate["bbox"]

            # Pastikan coordinates dalam boundary frame
            h, w = original_frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            if x2 > x1 and y2 > y1:
                self.captured_plate_image = original_frame[y1:y2, x1:x2].copy()

                # Timestamp
                self.capture_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Perform OCR
                self.ocr_text, self.ocr_confidence = self.ocr_reader.read_text(
                    self.captured_plate_image
                )

                print(
                    f"✅ Captured: '{self.ocr_text}' (Confidence: {self.ocr_confidence:.2f})"
                )

                # Save captured plate image dan log ke CSV
                if self.ocr_text and len(self.ocr_text) >= 3:
                    filename = f"captured_plate_{self.capture_time.replace(':', '-').replace(' ', '_')}.jpg"
                    filepath = os.path.join(config.OUTPUT_DIR, filename)
                    cv2.imwrite(filepath, self.captured_plate_image)
                    print(f"💾 Saved: {filepath}")

                    # ✅ LOG KE CSV
                    self.log_capture(self.ocr_text, self.ocr_confidence, filename)
            else:
                print("⚠️  Invalid plate coordinates for capture!")

        except Exception as e:
            print(f"❌ Error during capture: {e}")

    def process_live_video(self):
        """
        Main live video processing loop dengan resizable window
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Cannot open webcam")
            return

        # Set webcam resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("🎥 Starting live ALPR system...")
        print("   Press 'C' - Capture & Recognize plate")
        print("   Press 'ESC' or Close Window - Quit application")
        print("   💡 Window is resizable - drag edges to resize!")
        print(f"   📝 Log file: {self.log_file}")

        # CREATE RESIZABLE WINDOW
        window_name = "ALPR System - Live License Plate Recognition"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.current_width, self.current_height)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Cannot read frame from webcam")
                break

            # Detect plates in frame
            plates = self.plate_detector.detect_plates(frame, config.DETECTION_CONFIDENCE_THRESHOLD)

            # Store last detection untuk capture
            if plates:
                self.last_detection = plates[0]

            # GET CURRENT WINDOW SIZE
            try:
                window_size = cv2.getWindowImageRect(window_name)
                if window_size[2] > 0 and window_size[3] > 0:
                    self.current_width = max(self.min_width, window_size[2])
                    self.current_height = max(self.min_height, window_size[3])
            except:
                pass

            # Create dashboard dengan current size
            dashboard = self.create_dashboard(frame, plates)

            # Display instructions
            instruction_text = "Press 'C' to Capture | 'ESC' or Close Window to Quit"
            instruction_scale = min(0.6, 0.5 * (self.current_width / 800))
            cv2.putText(
                dashboard,
                instruction_text,
                (10, self.current_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                instruction_scale,
                (255, 255, 255),
                2,
            )

            # Show dashboard
            cv2.imshow(window_name, dashboard)

            # Handle keypress
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c") or key == ord("C"):
                if self.last_detection:
                    self.capture_and_recognize(self.last_detection, frame)
                else:
                    print("⚠️  No plate detected to capture!")
            elif key == 27:  # ESC key
                print("⎋ ESC pressed - Exiting...")
                break

            # CHECK: Jika window ditutup (tombol X)
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("❌ Window closed - Exiting...")
                break

        cap.release()
        cv2.destroyAllWindows()
        print("👋 ALPR System closed")


def main():
    """
    Main function untuk ALPR Live System
    """
    try:
        alpr_system = ALPRLiveSystem()
        alpr_system.process_live_video()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure webcam is connected and accessible")


if __name__ == "__main__":
    main()
