# main.py
import cv2
import os
import sys
import time

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

from utils.plate_detector import PlateDetector
from utils.ocr_reader import OCRReader
import config


class ALPRSystem:
    def __init__(self):
        """
        Initialize ALPR System
        """
        print("🚀 Initializing ALPR System...")

        # Initialize components
        self.plate_detector = PlateDetector(config.PLATE_DETECTOR_MODEL)
        self.ocr_reader = OCRReader()

        print("✅ ALPR System Ready!")

    def _bboxes_similar(self, bbox1, bbox2, threshold=0.7):
        """
        Check if two bounding boxes are similar based on IoU (Intersection over Union)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection area
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area

        # Calculate IoU
        iou = inter_area / union_area if union_area > 0 else 0

        return iou >= threshold

    def draw_modern_bounding_boxes(self, image, plates):
        """
        Draw modern bounding boxes with labels (similar to alpr_live.py)
        """
        result_image = image.copy()

        for plate in plates:
            x1, y1, x2, y2 = plate["bbox"]
            conf = plate["confidence"]

            # Pastikan coordinates valid untuk drawing
            if x2 > x1 and y2 > y1 and (x2 - x1) > 5 and (y2 - y1) > 5:
                try:
                    # 🎨 COLOR SETTINGS
                    color = (0, 210, 0)  # 🟢 HIJAU TOSCA
                    box_thickness = 3  # 📏 Medium thickness
                    font_scale = 0.6  # 🔤 Normal font size
                    text_thickness = 2  # 📝 Medium text thickness
                    bg_color = (0, 210, 0)  # 🟢 BACKGROUND SAMA
                    text_color = (255, 255, 255)  # ⚪ White text

                    # 1. Draw main bounding box
                    cv2.rectangle(
                        result_image,
                        (x1, y1),
                        (x2, y2),
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
                    bg_x1 = x1
                    bg_y1 = max(0, y1 - text_height - 10)  # 10px margin atas
                    bg_x2 = x1 + text_width + 10  # 10px padding kanan-kiri
                    bg_y2 = y1

                    # Pastikan background tidak keluar dari frame
                    if bg_y1 < 0:
                        bg_y1 = y1
                        bg_y2 = y1 + text_height + 10

                    # 5. Draw filled background
                    cv2.rectangle(
                        result_image,
                        (bg_x1, bg_y1),
                        (bg_x2, bg_y2),
                        bg_color,
                        -1,  # Filled rectangle
                    )

                    # 6. Calculate text position
                    text_x = bg_x1 + 5  # 5px padding dari kiri background
                    text_y = bg_y2 - 5  # 5px dari bawah background

                    # Jika background di atas, adjust text position
                    if bg_y1 < y1:
                        text_y = bg_y2 - 5
                    else:
                        text_y = bg_y1 + text_height + 5

                    # 7. Draw white text
                    cv2.putText(
                        result_image,
                        label_text,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        text_color,
                        text_thickness,
                    )

                    # 8. Draw OCR result below bounding box (kiri bawah)
                    plate_text = plate.get("ocr_text", "")
                    if plate_text:
                        # Background untuk OCR result
                        ocr_bg_x1 = x1
                        ocr_bg_y1 = y2 + 5
                        ocr_bg_x2 = x1 + 150  # Fixed width untuk OCR text
                        ocr_bg_y2 = y2 + 35

                        cv2.rectangle(
                            result_image,
                            (ocr_bg_x1, ocr_bg_y1),
                            (ocr_bg_x2, ocr_bg_y2),
                            (0, 0, 0),  # Black background
                            -1,
                        )

                        # OCR text
                        cv2.putText(
                            result_image,
                            f"Plate: {plate_text}",
                            (x1 + 5, y2 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),  # White text
                            1,
                        )

                except Exception as e:
                    print(f"⚠️  Error drawing bounding box: {e}")
                    # Fallback: simple rectangle
                    cv2.rectangle(
                        result_image,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

        return result_image

    def process_image(self, image_path, save_output=True):
        """
        Process single image for license plate recognition
        """
        print(f"📷 Processing image: {image_path}")

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error loading image: {image_path}")
            return None

        # Detect license plates
        plates = self.plate_detector.detect_plates(
            image, config.DETECTION_CONFIDENCE_THRESHOLD
        )

        print(f"🔍 Found {len(plates)} license plate(s)")

        results = []

        # Process each detected plate
        for i, plate in enumerate(plates):
            plate_img = plate["image"]
            bbox = plate["bbox"]
            det_conf = plate["confidence"]

            # OCR on plate
            plate_text, ocr_confidence = self.ocr_reader.read_text(plate_img)

            # Add OCR result to plate data for drawing
            plate["ocr_text"] = plate_text
            plate["ocr_confidence"] = ocr_confidence

            result = {
                "plate_id": i + 1,
                "text": plate_text,
                "detection_confidence": det_conf,
                "ocr_confidence": ocr_confidence,
                "bbox": bbox,
            }

            results.append(result)

            print(
                f"   🚗 Plate {i+1}: '{plate_text}' "
                f"(Det: {det_conf:.2f}, OCR: {ocr_confidence:.2f})"
            )

            # Save individual plate image
            if save_output:
                plate_filename = f"plate_{i+1}_{os.path.basename(image_path)}"
                plate_path = os.path.join(config.OUTPUT_DIR, plate_filename)
                cv2.imwrite(plate_path, plate_img)
                print(f"   💾 Saved plate image: {plate_filename}")

        # Draw modern bounding boxes on original image
        if save_output and plates:
            output_image = self.draw_modern_bounding_boxes(image, plates)

            output_path = os.path.join(
                config.OUTPUT_DIR, f"result_{os.path.basename(image_path)}"
            )
            cv2.imwrite(output_path, output_image)
            print(f"💾 Saved result image: {output_path}")

        return results

    def process_video(self, video_path, output_path=None, max_frames=None):
        """
        Process video for license plate recognition
        """
        print(f"🎥 Processing video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error opening video: {video_path}")
            return

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        detected_plates = []
        last_ocr_time = 0
        ocr_interval = 1.0  # OCR setiap 1 detik
        previous_plates = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = time.time()

            # Process frames
            plates = self.plate_detector.detect_plates(
                frame, config.DETECTION_CONFIDENCE_THRESHOLD
            )

            for plate in plates:
                plate["ocr_text"] = ""
                plate["ocr_confidence"] = 0.0

            # Perform OCR every 1 second for performance
            if current_time - last_ocr_time >= ocr_interval:
                plates_with_ocr = []
                for plate in plates:
                    plate_img = plate["image"]
                    plate_text, ocr_conf = self.ocr_reader.read_text(plate_img)

                    if plate_text:
                        plate["ocr_text"] = plate_text
                        plate["ocr_confidence"] = ocr_conf

                        detected_plates.append(
                            {
                                "frame": frame_count,
                                "text": plate_text,
                                "ocr_confidence": ocr_conf,
                                "detection_confidence": plate["confidence"],
                            }
                        )

                        print(f"Frame {frame_count}: Plate '{plate_text}'")

                    plates_with_ocr.append(plate)

                previous_plates = plates_with_ocr
                last_ocr_time = current_time
            else:
                # Gunakan hasil OCR sebelumnya
                for i, current_plate in enumerate(plates):
                    current_bbox = current_plate["bbox"]

                    for prev_plate in previous_plates:
                        prev_bbox = prev_plate["bbox"]

                        if self._bboxes_similar(current_bbox, prev_bbox):
                            current_plate["ocr_text"] = prev_plate.get("ocr_text", "")
                            current_plate["ocr_confidence"] = prev_plate.get(
                                "ocr_confidence", 0.0
                            )
                            break

            # Draw detections on frame
            frame_with_detections = self.draw_modern_bounding_boxes(frame, plates)

            # Write frame to output video
            if output_path:
                out.write(frame_with_detections)

            # Show preview (optional)
            cv2.imshow("Video Processing - Press ESC to stop", frame_with_detections)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break

            if max_frames and frame_count >= max_frames:
                break

        cap.release()
        if output_path:
            out.release()
            print(f"💾 Saved output video: {output_path}")

        cv2.destroyAllWindows()

        return detected_plates

    def process_live_camera(self):
        """
        Process live camera feed for license plate recognition
        """
        print("📹 Starting live camera... Press ESC to quit")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Cannot open webcam")
            return

        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        frame_count = 0
        last_ocr_time = 0
        ocr_interval = 1.0  # OCR setiap 1 detik

        # ✅ TAMBAHKAN: Variabel untuk menyimpan hasil OCR terakhir
        previous_plates = []

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error reading frame from camera")
                break

            frame_count += 1
            current_time = time.time()

            # Detect plates in every frame
            plates = self.plate_detector.detect_plates(
                frame, config.DETECTION_CONFIDENCE_THRESHOLD
            )

            # ✅ PERBAIKI: Gunakan hasil OCR sebelumnya sebagai default
            for plate in plates:
                plate["ocr_text"] = ""  # Default kosong
                plate["ocr_confidence"] = 0.0

            # Perform OCR every 1 second for performance
            if current_time - last_ocr_time >= ocr_interval:
                plates_with_ocr = []
                for plate in plates:
                    plate_img = plate["image"]
                    plate_text, ocr_conf = self.ocr_reader.read_text(plate_img)

                    if plate_text:
                        plate["ocr_text"] = plate_text
                        plate["ocr_confidence"] = ocr_conf
                        print(f"Frame {frame_count}: Detected Plate '{plate_text}'")

                    plates_with_ocr.append(plate)

                # ✅ PERBAIKI: Simpan plates dengan OCR untuk frame berikutnya
                previous_plates = plates_with_ocr
                last_ocr_time = current_time
            else:
                # ✅ PERBAIKI: Gunakan hasil OCR dari frame sebelumnya
                # Cari plate yang sesuai berdasarkan posisi bbox
                for i, current_plate in enumerate(plates):
                    current_bbox = current_plate["bbox"]

                    # Cari plate dengan bbox yang mirip dari previous_plates
                    for prev_plate in previous_plates:
                        prev_bbox = prev_plate["bbox"]

                        # Simple bbox matching berdasarkan overlap
                        if self._bboxes_similar(current_bbox, prev_bbox):
                            current_plate["ocr_text"] = prev_plate.get("ocr_text", "")
                            current_plate["ocr_confidence"] = prev_plate.get(
                                "ocr_confidence", 0.0
                            )
                            break

            # Draw modern bounding boxes
            frame_with_detections = self.draw_modern_bounding_boxes(frame, plates)

            # Display frame
            cv2.imshow("ALPR Live Camera - Press ESC to quit", frame_with_detections)

            # Check for ESC key or window close
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            if (
                cv2.getWindowProperty(
                    "ALPR Live Camera - Press ESC to quit", cv2.WND_PROP_VISIBLE
                )
                < 1
            ):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("👋 Live camera stopped")


def main():
    """
    Main function to run ALPR system with menu
    """
    try:
        # Initialize system
        alpr = ALPRSystem()

        while True:
            print("\n" + "=" * 50)
            print("🚀 ALPR SYSTEM - MAIN MENU")
            print("=" * 50)
            print("1. 📷 Process Image")
            print("2. 🎥 Process Video")
            print("3. 📹 Live Camera")
            print("0. ❌ Exit")
            print("-" * 50)

            choice = input("Select option (0-3): ").strip()

            if choice == "0":
                print("👋 Thank you for using ALPR System!")
                break

            elif choice == "1":
                # Image processing
                image_path = input(
                    "Enter image path (or press Enter for default 'images/gwalk.jpg'): "
                ).strip()
                if not image_path:
                    image_path = "images/gwalk.jpg"

                if os.path.exists(image_path):
                    results = alpr.process_image(image_path)
                    if results:
                        print("\n📊 Results:")
                        for result in results:
                            print(
                                f"   Plate: {result['text']} "
                                f"(Detection: {result['detection_confidence']:.2f}, "
                                f"OCR: {result['ocr_confidence']:.2f})"
                            )
                else:
                    print(f"❌ Image not found: {image_path}")
                    print("💡 Please check the file path and try again")

            elif choice == "2":
                # Video processing
                video_path = input("Enter video path: ").strip()
                if not video_path:
                    print("❌ Please enter a valid video path")
                    continue

                if os.path.exists(video_path):
                    output_path = os.path.join(
                        config.OUTPUT_DIR, f"processed_{os.path.basename(video_path)}"
                    )
                    results = alpr.process_video(video_path, output_path)
                    if results:
                        print(f"\n📊 Processed {len(results)} plates in video")
                else:
                    print(f"❌ Video not found: {video_path}")

            elif choice == "3":
                # Live camera
                print("🎥 Starting live camera detection...")
                alpr.process_live_camera()

            else:
                print("❌ Invalid option! Please enter 0, 1, 2, or 3")

            # Small pause before showing menu again
            input("\nPress Enter to continue...")

    except Exception as e:
        print(f"❌ Program error: {e}")
        print("💡 Check if all dependencies are installed and models are available")


if __name__ == "__main__":
    main()
