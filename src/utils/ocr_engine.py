import pytesseract
import cv2

def extract_text(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Optional: preprocess
    text = pytesseract.image_to_string(gray)
    return text.strip()
