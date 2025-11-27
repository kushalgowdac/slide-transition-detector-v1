import cv2
import pytesseract
import logging

# You can also set the Tesseract path here if needed
pytesseract.pytesseract.tesseract_cmd = r"D:\Programs\Tesseract-OCR\tesseract.exe"

def extract_text(image_path):
    if not os.path.exists(image_path):
        logging.warning(f"OCR skipped: File does not exist - {image_path}")
        return ""

    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image could not be read (None returned)")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Optional preprocessing (uncomment if needed)
        # gray = cv2.medianBlur(gray, 3)
        # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 10)

        text = pytesseract.image_to_string(gray)
        return text.strip()
    except Exception as e:
        logging.error(f"OCR failed on {image_path}: {e}")
        return ""
