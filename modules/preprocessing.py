import cv2
import numpy as np


class Preprocessor:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))

    def process(self, image):
        """Pipeline preprocessing untuk plat nomor"""
        try:
            # Convert to grayscale jika masih RGB
            if len(image.shape) == 3:
                gray = self.average_grayscale(image)
            else:
                gray = image

            # CLAHE contrast enhancement
            clahe_enhanced = self.apply_clahe(gray)

            # Gamma correction
            gamma_corrected = self.gamma_correction(clahe_enhanced)

            # NLM Denoising
            nlm_denoised = self.nlm_denoise(gamma_corrected)

            # Final contrast enhancement
            final_image = cv2.convertScaleAbs(nlm_denoised, alpha=1.5, beta=0)

            return final_image

        except Exception as e:
            # print(f"❌ Preprocessing error: {str(e)}")
            # Return original image jika error
            return image

    def average_grayscale(self, image):
        """Convert ke grayscale menggunakan average method"""
        try:
            if len(image.shape) == 3:
                # Normalize ke 0-255 dulu
                if image.dtype != np.uint8:
                    image = np.uint8(image)

                blue = image[:, :, 0].astype(float)
                green = image[:, :, 1].astype(float)
                red = image[:, :, 2].astype(float)
                gray = (blue + green + red) / 3.0
                return np.uint8(np.clip(gray, 0, 255))
            else:
                return np.uint8(image)
        except Exception as e:
            # print(f"❌ Grayscale error: {str(e)}")
            return image

    def apply_clahe(self, grayscale_image):
        """Apply CLAHE contrast enhancement"""
        try:
            # Pastikan format uint8 dan 2D
            if grayscale_image.dtype != np.uint8:
                grayscale_image = np.uint8(grayscale_image)

            if len(grayscale_image.shape) > 2:
                grayscale_image = grayscale_image[:, :, 0]

            return self.clahe.apply(grayscale_image)
        except Exception as e:
            # print(f"❌ CLAHE error: {str(e)}")
            return grayscale_image

    def gamma_correction(self, image, gamma=1.3):
        """Apply gamma correction"""
        try:
            # Pastikan format uint8 dan 2D
            if image.dtype != np.uint8:
                image = np.uint8(image)

            if len(image.shape) > 2:
                image = image[:, :, 0]

            inv_gamma = 1.0 / gamma
            table = np.array(
                [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
            ).astype("uint8")
            return cv2.LUT(image, table)
        except Exception as e:
            # print(f"❌ Gamma correction error: {str(e)}")
            return image

    def nlm_denoise(self, image, h=15, template_window=7, search_window=21):
        """Apply Non-Local Means Denoising"""
        try:
            # Pastikan format uint8 dan 2D
            if image.dtype != np.uint8:
                image = np.uint8(image)

            if len(image.shape) > 2:
                image = image[:, :, 0]

            return cv2.fastNlMeansDenoising(
                image, None, h, template_window, search_window
            )
        except Exception as e:
            # print(f"❌ NLM denoise error: {str(e)}")
            return image
