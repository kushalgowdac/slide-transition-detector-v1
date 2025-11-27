import cv2
import numpy as np
import os
import logging

def detect_transitions(video_path, output_dir, threshold=0.1, frame_skip=30):
    logging.info(f"Opening video: {video_path}")
    if not os.path.exists(video_path):
        logging.error(f"Video file does not exist: {video_path}")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Failed to open video.")
        return []

    prev_hist = None
    slide_num = 0
    timestamps = []

    os.makedirs(output_dir, exist_ok=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if frame_id % frame_skip != 0:
                continue

            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()

                if prev_hist is not None:
                    diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                    if diff > threshold:
                        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                        filename = f"slide_{slide_num:03}.jpg"
                        cv2.imwrite(os.path.join(output_dir, filename), frame)
                        timestamps.append((filename, round(timestamp, 2)))
                        logging.info(f"Detected slide: {filename} at {timestamp:.2f}s")
                        slide_num += 1

                prev_hist = hist
            except Exception as e:
                logging.warning(f"Failed to process frame {frame_id}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error while reading video: {e}")
    finally:
        cap.release()

    return timestamps
