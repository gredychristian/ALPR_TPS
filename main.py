# main.py
import cv2
import os
import sys

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

        # Draw detections on original image
        if save_output and plates:
            output_image = self.plate_detector.draw_detections(image, plates)

            # Add OCR results as text
            for result in results:
                x1, y1, x2, y2 = result["bbox"]
                text = f"{result['text']} ({result['ocr_confidence']:.2f})"
                cv2.putText(
                    output_image,
                    text,
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

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

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Process every 10th frame for performance
            if frame_count % 10 == 0:
                plates = self.plate_detector.detect_plates(
                    frame, config.DETECTION_CONFIDENCE_THRESHOLD
                )

                for plate in plates:
                    plate_img = plate["image"]
                    plate_text, ocr_conf = self.ocr_reader.read_text(plate_img)

                    if plate_text:  # Only add if OCR found text
                        detected_plates.append(
                            {
                                "frame": frame_count,
                                "text": plate_text,
                                "ocr_confidence": ocr_conf,
                                "detection_confidence": plate["confidence"],
                            }
                        )

                        print(f"Frame {frame_count}: Plate '{plate_text}'")

                # Draw detections on frame
                frame_with_detections = self.plate_detector.draw_detections(
                    frame, plates
                )

                # Write frame to output video
                if output_path:
                    out.write(frame_with_detections)

            if max_frames and frame_count >= max_frames:
                break

        cap.release()
        if output_path:
            out.release()
            print(f"💾 Saved output video: {output_path}")

        return detected_plates


def test_with_webcam(alpr_system):
    """
    Test ALPR with webcam in real-time
    """
    cap = cv2.VideoCapture(0)  # Webcam
    print("📹 Starting webcam... Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect plates every 5 frames untuk performance
        plates = alpr_system.plate_detector.detect_plates(frame)

        # Draw detections
        frame_with_detections = alpr_system.plate_detector.draw_detections(
            frame, plates
        )

        # OCR pada plate yang terdeteksi
        for plate in plates:
            plate_text, ocr_conf = alpr_system.ocr_reader.read_text(plate["image"])
            if plate_text:
                x1, y1, x2, y2 = plate["bbox"]
                cv2.putText(
                    frame_with_detections,
                    f"OCR: {plate_text}",
                    (x1, y2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

        cv2.imshow("ALPR System - Webcam", frame_with_detections)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    """
    Main function to run ALPR system
    """
    try:
        # Initialize system
        alpr = ALPRSystem()

        # TEST 1: Process single image
        image_path = "images/gwalk.jpg"

        if os.path.exists(image_path):
            results = alpr.process_image(image_path)
            print("\n📊 Results:")
            for result in results:
                print(
                    f"   Plate: {result['text']} "
                    f"(Detection: {result['detection_confidence']:.2f}, "
                    f"OCR: {result['ocr_confidence']:.2f})"
                )
        else:
            print(f"❌ Test image not found: {image_path}")
            print("💡 Please add car images to 'images/' folder first!")

        # TEST 2: Webcam option
        use_webcam = input("🚗 Want to test with webcam? (y/n): ")
        if use_webcam.lower() == "y":
            test_with_webcam(alpr)

    except Exception as e:
        print(f"❌ Program error: {e}")
        print("💡 Check if all dependencies are installed and models are available")


if __name__ == "__main__":
    main()
