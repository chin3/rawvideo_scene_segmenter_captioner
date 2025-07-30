import argparse
import json
from utils.video import extract_frames
from utils.blip import caption_image
from utils.similarity import caption_similarity, relevance_to_goal

def run_ingestion(video_path, goal=None, interval=5, similarity_threshold=0.75):
    frames = extract_frames(video_path, interval_sec=interval)
    captions = [caption_image(path) for _, path in frames]

    segments = []
    prev_caption = captions[0]
    start_time = frames[0][0]

    for i in range(1, len(frames)):
        current_time, frame_path = frames[i]
        similarity = caption_similarity(prev_caption, captions[i])
        
        if similarity < similarity_threshold:
            segment = {
                "start": start_time,
                "end": current_time,
                "caption": prev_caption,
                "key_frame": frames[i-1][1]
            }
            if goal:
                segment["relevance"] = relevance_to_goal(prev_caption, goal)
            segments.append(segment)
            start_time = current_time
            prev_caption = captions[i]
    
    # Add final segment
    segments.append({
        "start": start_time,
        "end": frames[-1][0],
        "caption": captions[-1],
        "key_frame": frames[-1][1],
        "relevance": relevance_to_goal(captions[-1], goal) if goal else None
    })

    # Optional filtering
    if goal:
        segments = [s for s in segments if s["relevance"] > 0.5]

    with open("metadata.json", "w") as f:
        json.dump(segments, f, indent=2)
    
    print(f"[✔] Saved {len(segments)} segments to metadata.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--goal", help="Optional prompt to focus on")
    parser.add_argument("--frame_interval", type=int, default=5)
    args = parser.parse_args()

    run_ingestion(args.video_path, goal=args.goal, interval=args.frame_interval)
