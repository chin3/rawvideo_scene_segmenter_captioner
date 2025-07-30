import cv2
import os

def extract_frames(video_path, interval_sec=5, output_dir="frames"):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    timestamps = []
    current_frame = 0
    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        success, frame = cap.read()
        if not success:
            break
        seconds = int(current_frame / fps)
        filename = os.path.join(output_dir, f"frame_{seconds:04d}.jpg")
        cv2.imwrite(filename, frame)
        timestamps.append((seconds, filename))
        current_frame += int(fps * interval_sec)
    
    cap.release()
    return timestamps
