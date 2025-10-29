import os
import json
import time
import cv2
import warnings
from datetime import datetime
from modules.plat_detector import PlatDetector
from modules.char_detector import CharDetector
from modules.preprocessing import Preprocessor

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class ALPRSystem:
    def __init__(self):
        # Path configuration
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.base_dir, "models")
        self.data_dir = os.path.join(self.base_dir, "data")

        # Model paths
        self.plat_model_path = os.path.join(self.models_dir, "plat.pt")
        self.char_model_path = os.path.join(self.models_dir, "char.pt")

        # Data paths
        self.input_dir = os.path.join(self.data_dir, "input")
        self.output_dir = os.path.join(self.data_dir, "output")
        self.json_dir = os.path.join(self.data_dir, "json")

        # Create directories if not exist
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)

        # Initialize components
        print("ALPR TPS 2025")
        self.plat_detector = PlatDetector(self.plat_model_path)
        self.char_detector = CharDetector(self.char_model_path, conf_threshold=0.2)
        self.preprocessor = Preprocessor()
        # print("✅ Models loaded successfully")

        # JSON file
        self.json_file = os.path.join(self.json_dir, "vehicle_log.json")

        # Initialize JSON file if not exists or corrupt
        self._initialize_json_file()

        # Initialize counter from existing data
        self.counter = self._get_next_id()

    def _initialize_json_file(self):
        """Initialize JSON file or fix if corrupt"""
        try:
            # Cek jika file exists dan valid
            if os.path.exists(self.json_file):
                with open(self.json_file, "r") as f:
                    json.load(f)  # Coba parse JSON
                # print("✅ JSON file is valid")
            else:
                # Buat file baru jika tidak exists
                with open(self.json_file, "w") as f:
                    json.dump([], f)
                # print("✅ Created new JSON file")

        except json.JSONDecodeError:
            # Jika JSON corrupt, buat baru
            # print("⚠️  JSON file corrupt, creating new one...")
            with open(self.json_file, "w") as f:
                json.dump([], f)
            # print("✅ Fixed JSON file")

    def _get_next_id(self):
        """Get next ID from existing JSON data"""
        try:
            with open(self.json_file, "r") as f:
                data = json.load(f)

            if not data:
                return 1  # Start from 1 if no data
            else:
                # Cari ID tertinggi
                max_id = max(item["id"] for item in data)
                return max_id + 1

        except Exception as e:
            # print(f"❌ Error reading JSON for ID: {e}")
            return 1  # Fallback ke 1

    def process_image(self, image_path):
        """Process single image"""
        try:
            # print(f"🚗 Processing image: {image_path}")

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                # print(f"❌ Gagal load gambar: {image_path}")
                return

            # print(f"📊 Original image - Shape: {image.shape}, dtype: {image.dtype}")

            # Simpan image asli untuk output (tanpa konversi RGB)
            original_image_bgr = image.copy()
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Hanya untuk processing

            # Generate ID (auto-increment dari JSON)
            vehicle_id = self.counter
            self.counter += 1

            # Detect plate
            cropped_plate, plate_bbox = self.plat_detector.detect(image_rgb)

            # Initialize status flags
            plat_detected = False
            char_detected = False
            plate_text = ""
            output_image = original_image_bgr.copy()  # Gunakan BGR untuk output

            if cropped_plate is not None:
                plat_detected = True  # ✅ Plat terdeteksi
                # print(f"✅ Plat terdeteksi - Crop shape: {cropped_plate.shape}")

                # Preprocess plate image
                processed_plate = self.preprocessor.process(cropped_plate)

                # Detect characters
                characters = self.char_detector.detect(processed_plate)
                plate_text = self.char_detector.get_plate_text(characters)

                if len(characters) > 0:
                    char_detected = True  # ✅ Karakter terdeteksi
                    # print(f"✅ Characters detected: {len(characters)}")
                else:
                    # print(f"❌ No characters detected")
                    pass

                    # Draw detections pada output image (masih BGR)
                output_image = self.plat_detector.draw_detection(
                    output_image, plate_bbox
                )

                # Draw characters pada processed plate (untuk output)
                if len(characters) > 0:
                    char_image = self.char_detector.draw_detections(
                        processed_plate, characters
                    )
                else:
                    char_image = processed_plate
            else:
                # print(f"❌ Plat tidak terdeteksi")
                # plat_detected tetap False
                # Jika plat tidak terdeteksi, tetap preprocessing gambar utama
                processed_plate = self.preprocessor.process(image_rgb)
                char_image = processed_plate

            # Save output images
            input_filename = os.path.basename(image_path)
            name, ext = os.path.splitext(input_filename)

            output_path = os.path.join(self.output_dir, f"{vehicle_id}_processed{ext}")
            plate_path = os.path.join(self.output_dir, f"{vehicle_id}_plate{ext}")

            # TIDAK PERLU KONVERSI - output_image sudah BGR
            cv2.imwrite(output_path, output_image)

            # Handle char_image format untuk saving
            if len(char_image.shape) == 2:  # Grayscale
                plate_image_bgr = cv2.cvtColor(char_image, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(plate_path, plate_image_bgr)
            else:  # RGB
                plate_image_bgr = cv2.cvtColor(char_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(plate_path, plate_image_bgr)

            # Save to JSON dengan status baru
            self.save_to_json(
                vehicle_id,
                plate_text,
                image_path,
                output_path,
                plate_path,
                plat_detected,
                char_detected,  # ⭐ TAMBAH PARAMETER INI
            )

            # Tampilkan status
            status_msg = (
                "FULL SUCCESS"
                if (plat_detected and char_detected)
                else "PLAT ONLY" if plat_detected else "FAILED"
            )
            print(
                # f"✅ Kendaraan {vehicle_id} diproses - Status: {status_msg} - Plat: '{plate_text if plate_text else 'Tidak Terbaca'}'"
            )

            # Hapus file input setelah sukses
            os.remove(image_path)
            # try:
            #     print(f"🗑️  Input file deleted: {os.path.basename(image_path)}")
            # except Exception as e:
            #     print(f"❌ Gagal hapus file input: {str(e)}")

        except Exception as e:
            # print(f"❌ Error processing {image_path}: {str(e)}")
            import traceback

            traceback.print_exc()


    def save_to_json(
        self,
        vehicle_id,
        plate_text,
        input_path,
        output_path,
        plate_path,
        plat_detected,
        char_detected,
    ):
        """Save data to JSON log"""
        try:
            # Baca data existing
            with open(self.json_file, "r") as f:
                data = json.load(f)

            # Gunakan path relative
            relative_output = os.path.relpath(output_path, self.base_dir)
            relative_plate = os.path.relpath(plate_path, self.base_dir)

            vehicle_data = {
                "id": vehicle_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "plate_number": plate_text,
                "output_image": relative_output,
                "plate_image": relative_plate,
                "plat_detected": plat_detected,
                "char_detected": char_detected,
                "status": (
                    "success"
                    if (plat_detected and char_detected)
                    else "partial" if plat_detected else "failed"
                ),
            }

            data.append(vehicle_data)

            # Tulis ulang dengan indent
            with open(self.json_file, "w") as f:
                json.dump(data, f, indent=2)

            # print(f"✅ Saved to JSON: ID {vehicle_id}")

            # TAMBAHKAN INI UNTUK PRINT JSON DI TERMINAL
            # print("📄 JSON Data:")
            print(json.dumps(vehicle_data, indent=2))

        except Exception as e:
            # print(f"❌ Error saving to JSON: {str(e)}")
            # Coba buat JSON baru jika corrupt
            self._initialize_json_file()

    def monitor_input_folder(self):
        """Monitor input folder for new images"""
        # print("🚀 ALPR System Started...")
        # print(f"📁 Monitoring folder: {self.input_dir}")
        # print(f"📊 Next ID: {self.counter}")
        print("ALPR Running...")
        print("Press Ctrl+C to stop")

        processed_files = set()

        try:
            while True:
                # Check for new images
                for filename in os.listdir(self.input_dir):
                    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        file_path = os.path.join(self.input_dir, filename)

                        if file_path not in processed_files:
                            print(f"📸 New image detected: {filename}")
                            self.process_image(file_path)
                            processed_files.add(file_path)

                time.sleep(1)  # Check every 1 second

        except KeyboardInterrupt:
            print("\nALPR Stopped")


if __name__ == "__main__":
    alpr_system = ALPRSystem()
    alpr_system.monitor_input_folder()
