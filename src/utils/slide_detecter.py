import cv2
import numpy as np
import os

def detect_transitions(video_path, output_dir, threshold=0.5, frame_skip=30):
    cap = cv2.VideoCapture(video_path)
    prev_hist = None
    slide_num = 0
    timestamps = []

    os.makedirs(output_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_id % frame_skip != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0,256])
        hist = cv2.normalize(hist, hist).flatten()

        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if diff > threshold:
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                filename = f"slide_{slide_num:03}.jpg"
                cv2.imwrite(os.path.join(output_dir, filename), frame)
                timestamps.append((filename, round(timestamp, 2)))
                slide_num += 1

        prev_hist = hist

    cap.release()
    return timestamps
