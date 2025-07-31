import os
import json
from utils.blip import caption_image
from utils.similarity import caption_similarity, relevance_to_goal
import pandas as pd

def load_existing_frames(folder="frames"):
    frames = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".jpg"):
            seconds = int(filename.split("_")[1].split(".")[0])
            path = os.path.join(folder, filename)
            frames.append((seconds, path))
    return frames

def process_existing_frames(goal=None, similarity_threshold=0.75):
    frames = load_existing_frames()
    captions = []

    for i, (_, path) in enumerate(frames):
        print(f"[🔎] Captioning {i+1}/{len(frames)}: {path}")
        caption = caption_image(path)
        captions.append(caption)

    segments = []
    prev_caption = captions[0]
    start_time = frames[0][0]

    for i in range(1, len(frames)):
        current_time, _ = frames[i]
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

    # Final segment
    segments.append({
        "start": start_time,
        "end": frames[-1][0],
        "caption": captions[-1],
        "key_frame": frames[-1][1],
        "relevance": relevance_to_goal(captions[-1], goal) if goal else None
    })

    if goal:
        segments = [s for s in segments if s["relevance"] > 0.5]

    with open("metadata.json", "w") as f:
        json.dump(segments, f, indent=2)
    df = pd.DataFrame(segments)
    df.to_csv("scene_metadata.csv", index=False)
    print(f"[📄] Also saved scene metadata to scene_metadata.xlsx")

    print(f"[✅] Wrote {len(segments)} segments to metadata.json")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", help="Optional goal prompt", default=None)
    args = parser.parse_args()

    process_existing_frames(goal=args.goal)
