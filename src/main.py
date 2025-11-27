import os
import json
import logging

from src.utils.slide_detector import detect_transitions
from src.utils.ocr_engine import extract_text

# ─── Configure Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# ─── File Paths ──────────────────────────────────────────────────────
video_path = "data/input_video.mp4"
slide_dir = "output/slides"
results_file = "output/results.json"

try:
    logging.info("Starting slide detection...")
    transitions = detect_transitions(video_path, slide_dir)
    logging.info(f"Found {len(transitions)} transitions.")
except Exception as e:
    logging.error(f"Failed during slide detection: {e}")
    transitions = []

output = []
for filename, timestamp in transitions:
    try:
        full_path = os.path.join(slide_dir, filename)
        logging.info(f"OCR on {filename} at {timestamp}s")
        text = extract_text(full_path)
        output.append({
            "filename": filename,
            "timestamp": timestamp,
            "text": text[:200] + "..." if len(text) > 200 else text
        })
    except Exception as e:
        logging.warning(f"OCR failed on {filename}: {e}")

try:
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(output, f, indent=4)
    logging.info(f"Results saved to {results_file}")
except Exception as e:
    logging.error(f"Failed to save results: {e}")
