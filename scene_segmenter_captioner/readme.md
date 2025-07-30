
# 🧠 `scene_segmenter_captioner` – Hybrid Scene Segmenter & Captioner

This project extracts **keyframes** from a video, generates **BLIP-2 captions**, and segments the video into **semantic scenes** based on **caption similarity** — optionally scoring each scene by a user-defined **goal prompt** (e.g. *"focus on tigers attacking prey"*).

---

## ✅ Current Features (MVP Complete)

### 🎞️ Frame Extraction (`utils/video.py`)

* Samples frames from any `.mp4` video at a user-defined interval (e.g. every 5 seconds).
* Stores them in `frames/` as `frame_0010.jpg`, etc.

### 🧠 Captioning with BLIP (`utils/blip.py`)

* Uses `Salesforce/blip-image-captioning-base` to generate captions from images.
* Captions are used for semantic segmentation and optional filtering.

### 🔍 Semantic Segmentation (`utils/similarity.py`)

* Uses `sentence-transformers` to compute similarity between consecutive captions.
* Marks a **new scene** when similarity drops below a threshold (default: 0.75).
* Optionally filters or scores scenes using a **goal prompt**.

### 📄 JSON Output

Outputs `metadata.json` containing a list of scene segments:

```json
[
  {
    "start": 10,
    "end": 25,
    "caption": "A tiger walking toward the camera",
    "key_frame": "frames/frame_0010.jpg",
    "relevance": 0.88
  }
]
```

---

## 🧪 How to Run

### Full pipeline (extract frames + caption + segment):

```bash
python ingest.py --video_path tiger.mp4 --goal "focus on tigers attacking prey" --frame_interval 5
```

### Just run captioning/segmentation on **existing frames**:

```bash
python caption_only.py --goal "focus on tigers attacking prey"
```

> Use this when you've already generated the frames in `frames/` and want to resume without rerunning the full video load.

---

## 🧰 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

```txt
opencv-python
transformers
torch
sentence-transformers
Pillow
```

---

## 🧱 Folder Structure

```
video_ingestor/
├── ingest.py             # Full pipeline
├── caption_only.py       # Resume-only script
├── utils/
│   ├── video.py          # Frame sampling
│   ├── blip.py           # Image captioning
│   └── similarity.py     # Embedding + semantic comparison
├── frames/               # Saved .jpg keyframes
├── metadata.json         # Final segmented scene info
├── requirements.txt
```

---
## Next Steps 

* Support for export to CSV for metadata.json
* Add timestamps to the data schema for video scrub support. 

## 🧩 Potential Next Steps


### 🖼️ 1. Add Frame Thumbnails or Previews

* Generate `.gif` or `.mp4` clips per segment
* Add links to frame previews in `metadata.json`

### ⚡ 2. Switch to Faster Captioning (BLIP-2 or LLaVA)

* Upgrade to `blip2-flan-t5-xl` or try `LLaVA` for richer captions
* Optimize with quantization if running locally

### 🧠 3. Use Your Own Goal Prompting Logic

* Add support for `--goal_file` or structured task templates
* Create multiple JSONs for different query prompts

### 📈 4. Add Progress Bar / Logging

* Use `tqdm` to show captioning progress
* Save logs to file for batch jobs

### 🌐 5. Build a Simple UI

* Flask or Streamlit viewer for segmented scenes
* Click-to-preview each frame and its caption/relevance

---

## 🧠 Remember

If this error comes up again:

```
AttributeError: 'FeatureExtractionOutput' object has no attribute 'input_ids'
```

✅ It can be fixed by making sure the inputs are mapped correctly to handle non video input:

```python
model.to(device)
inputs = processor(...).to(device)
```

For me:

start virtual env
.\venv\Scripts\Activate.ps1
install requirments
pip install -r requirements.txt