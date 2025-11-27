from src.utils.slide_detector import detect_transitions
from src.utils.ocr_engine import extract_text
import json
import os

video_path = "data/input_video.mp4"
slide_dir = "output/slides"
results_file = "output/results.json"

transitions = detect_transitions(video_path, slide_dir)

output = []
for filename, timestamp in transitions:
    text = extract_text(os.path.join(slide_dir, filename))
    output.append({
        "filename": filename,
        "timestamp": timestamp,
        "text": text[:200] + "..." if len(text) > 200 else text  # truncate for now
    })

with open(results_file, "w") as f:
    json.dump(output, f, indent=4)

print(f"Processed {len(output)} slides. Output written to {results_file}")
