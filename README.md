# Sign Language Data Pipeline

A pipeline for building large-scale sign language datasets from YouTube videos. It downloads videos, separates labeled (subtitled) from unlabeled footage, generates subtitles where missing, tracks people with YOLO, and extracts body/hand/face landmarks with **rtmlib Wholebody3d** (RTMW3D-x).

---

## Requirements

- Python 3.12
- `ffmpeg` on PATH (with CUDA support for GPU-accelerated video decoding in step 06)
- NVIDIA GPU recommended (rtmlib CUDA via onnxruntime-gpu in step 06)

Install dependencies:
```bash
uv sync
```

> **Note — CUDA version mismatch**: `onnxruntime-gpu` is built against CUDA 12. If your system has CUDA 13+, the `nvidia-cublas-cu12`, `nvidia-cufft-cu12`, and `nvidia-cuda-runtime-cu12` packages (included in `pyproject.toml`) provide the required CUDA 12 runtime libs. They are preloaded automatically at import time — no manual setup needed.

---

## Config file

Every step reads the same YAML config. Pass it with `--config path/to/config.yaml`. The working directory defaults to `./working` and can be overridden with `--workdir`.

Config is parsed into typed dataclasses in `utils/config.py`. **All defaults live there** — any key omitted from the YAML falls back to its default value.

```yaml
sources:
  - name: "source_a"
    url: "https://www.youtube.com/@SomeChannel"  # download entire channel
    subs: true          # channel has human-made subtitles
    auto_subs: false    # do not use YouTube auto-generated subs
    generate_subs: false

  - name: "source_b"
    path: "urls.txt"    # one URL per line; mutually exclusive with `url`
    subs: false
    auto_subs: false
    generate_subs: true # transcribe with Whisper (step 04)

  - name: "source_c"
    url: "https://www.youtube.com/@AnotherChannel"
    subs: false
    auto_subs: true     # accept YouTube auto-generated subs as labels
    generate_subs: false

options:
  download:
    format: "bestvideo[height<=720]+bestaudio/best[height<=720]"
    sub_langs: ["es.*"]
    cookies_from_browser: "firefox"   # optional

  video_audio_separation:
    delete_original: false

  whisper:
    model: "large-v3-turbo"
    device: "cuda"
    language: "es"

  bounding_boxes:
    model_path: "models/yolo11m.pt"   # relative to --workdir
    fps: 6
    batch_size: 32
    batch_queue: 32

  landmarks:
    fps: 6
    num_workers: 1
    max_clip_frame_separation: 1      # seconds; gap before a new clip is started
    min_clip_duration_frames: 36      # clips shorter than this are discarded (fps * 6s)
    moving_threshold: 0.25            # clips with less motion than this are discarded
    write_buffer_size: 160            # frames buffered before flushing to H5
    mode: "balanced"                  # rtmlib model size: balanced | performance | lightweight
    backend: "onnxruntime"            # onnxruntime | opencv | openvino | tensorrt
    device: "cuda"
```

### Source flags

| Flag | Meaning |
|------|---------|
| `subs: true` | Videos come with human-made subtitles. Videos with a matching `.vtt` → labeled; rest → unlabeled. |
| `auto_subs: true` | Accept YouTube's auto-generated subtitles as labels. Same sorting logic applies. |
| `generate_subs: true` | No downloaded subs. Step 03 runs VAD to filter silent videos; step 04 transcribes the rest with Whisper. |
| (none of the above) | All videos → unlabeled. |

---

## Running the pipeline

Run each step in order from the project root:

```bash
python pipeline/01_download.py                       --config config.yaml
python pipeline/02_separate_audio_video_subtitles.py --config config.yaml
python pipeline/03_parse_subtitles.py                --config config.yaml
python pipeline/04_generate_subs.py                  --config config.yaml  # only if generate_subs sources exist
python pipeline/05_generate_bounding_boxes.py        --config config.yaml
python pipeline/06_generate_landmarks.py             --config config.yaml
```

Skip step 04 entirely if no source has `generate_subs: true`.

---

## Pipeline steps

### Step 01 — Download (`01_download.py`)

Downloads videos from YouTube using `yt-dlp`. For each source, either a channel URL or a text file of URLs is accepted. Subtitle files (`.vtt`) are downloaded alongside the video when available.

Output: `working/videos/{source}/`

---

### Step 02 — Separate audio, video, subtitles (`02_separate_audio_video_subtitles.py`)

Splits each downloaded `.mp4` into:
- `video/` — video-only stream (no re-encode, container copy)
- `audio/` — mono 16 kHz MP3
- `subtitles/` — moves `.vtt` files here

Runs in parallel across all CPU cores. Optionally deletes the original merged file (`delete_original`).

---

### Step 03 — Sort labeled / unlabeled (`03_parse_subtitles.py`)

Decides whether each video is **labeled** (has a subtitle) or **unlabeled** (no subtitle), and moves files accordingly.

- **Sources with `subs` or `auto_subs`**: any video whose stem matches a `.vtt` in `subtitles/` is labeled; the rest are VAD-filtered (silero-VAD) and moved to `unlabeled/video/` if no speech is found.
- **Sources with `generate_subs`**: VAD-filters all videos; silent videos go to `unlabeled/video/`; the rest remain in `video/` for step 04.
- **Sources with none of the above**: everything goes to `unlabeled/video/`.

---

### Step 04 — Generate subtitles with Whisper (`04_generate_subs.py`)

Only runs for sources with `generate_subs: true`. Transcribes each video's audio using `faster-whisper` and writes a `.vtt` file, then moves the video to `labeled/video/`.

---

### Step 05 — Generate bounding boxes (`05_generate_bounding_boxes.py`)

Runs YOLO tracking (`ultralytics`) on every video (labeled and unlabeled) to detect and track people frame by frame. Frames are decoded with PyAV and fed to YOLO in batches via a producer–consumer thread pair.

Output per video:
```
working/processed/bounding_boxes/{video.mp4}/{video_stem}.json
```

JSON format:
```json
[
  {"timestamp": 0.167, "boxes": {"1.0": [x1, y1, x2, y2], "2.0": [...]}},
  ...
]
```

Track IDs are floats serialized as string keys after the JSON round-trip.

---

### Step 06 — Generate landmarks (`06_generate_landmarks.py`)

The main feature-extraction step. For each video it:

1. Loads the bounding box JSON from step 05 and groups detections into per-person clips (a new clip starts whenever a track disappears for more than `max_clip_frame_separation` seconds).
2. Reads the VTT subtitle file if labeled.
3. Reads the video once per person in a single sequential pass (`GPUVideoReader` via `ffmpeg -hwaccel cuda`, falling back to PyAV).
4. For each frame, passes the **full frame + pre-computed bounding box** from step 05 directly to the **rtmlib Wholebody3d pose model**, bypassing rtmlib's internal person detector.
5. Extracts 133 COCO-WholeBody keypoints (body 17, face 68, left hand 21, right hand 21) with absolute pixel coordinates and metric depth (z).
6. Discards static clips (torso vectors change less than `moving_threshold` over the first `min_clip_duration_frames` frames) and clips shorter than 1 second.
7. Writes landmarks to a per-video temp H5 file in chunks of `write_buffer_size` frames.
8. After all videos are processed, merges temp files into two source-level H5 files.

Parallelized across videos with `pathos.ProcessPool(nodes=num_workers)`.

#### Output files

```
working/processed/landmarks/{source}_labeled.h5
working/processed/landmarks/{source}_unlabeled.h5
```

#### H5 structure

```
{source}_labeled.h5
├── .attrs["fps"] = 6
└── {video_stem}/
    ├── .attrs["video_id"]   = str
    ├── .attrs["labeled"]    = bool
    ├── .attrs["subtitles"]  = str   (full VTT content; "" for unlabeled)
    └── person_{track_id}/
        └── {clip_index}/
            ├── .attrs["start"]             float  (seconds)
            ├── .attrs["end"]               float  (seconds)
            ├── body_landmarks              (N, 17, 3)  float64  — COCO 17 body keypoints
            ├── left_hand_landmarks         (N, 21, 3)  float64
            ├── right_hand_landmarks        (N, 21, 3)  float64
            ├── face_landmarks              (N, 68, 3)  float64
            └── timestamps                  (N,)        float64  (seconds)
```

Coordinates are in **absolute pixel space** (x, y in pixels of the original frame; z is metric depth in meters from rtmlib).

#### Keypoint layout (COCO-WholeBody, 133 total)

| Dataset | Indices | Count | Description |
|---------|---------|-------|-------------|
| `body_landmarks` | 0–16 | 17 | COCO body skeleton |
| *(feet, not stored)* | 17–22 | 6 | — |
| `face_landmarks` | 23–90 | 68 | Face mesh |
| `left_hand_landmarks` | 91–111 | 21 | Left hand |
| `right_hand_landmarks` | 112–132 | 21 | Right hand |

---

## Working directory layout

```
working/
├── videos/
│   └── {source}/
│       ├── labeled/
│       │   ├── video/        ← labeled .mp4 files
│       │   ├── audio/        ← .mp3 files
│       │   └── subtitles/    ← .vtt files
│       └── unlabeled/
│           ├── video/
│           └── audio/
└── processed/
    ├── bounding_boxes/
    │   └── {video.mp4}/
    │       └── {video_stem}.json
    └── landmarks/
        ├── tmp/
        │   └── {source}/
        │       └── {video_stem}.h5   ← intermediate per-video files
        ├── {source}_labeled.h5
        └── {source}_unlabeled.h5
```

---

## Dataset Statistics

You can calculate the current dataset statistics (number of videos and total duration) using the exploration script:

```bash
python utils/exploration/calculate_video_hours.py --config config.yaml
```

