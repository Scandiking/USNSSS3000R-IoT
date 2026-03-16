# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a Raspberry Pi waste sorting system (IoT course project, University of South-Eastern Norway). A camera captures waste items, a TensorFlow/LiteRT model classifies them into one of 6 categories, and GPIO-connected LEDs indicate the correct bin. A Flask server on Windows provides a web push notification dashboard.

## Key commands

**Train the model** (run on a PC with a GPU/decent CPU, not the Pi):
```bash
cd waste_classifier/dataset
python train_model.py
```
Outputs `waste_classifier_savedmodel/`, `waste_classifier_model.h5`, and `waste_classifier.tflite` in the same directory.

**Prepare dataset** (run once before training, if dataset has not been split yet):
```bash
cd waste_classifier/dataset
python split_dataset.py
```
Reads from `train/` and splits images 70/15/15 into `train/`, `validation/`, `test/` subdirectories.

**Run inference on Raspberry Pi** (uses PiCamera2 + GPIO LEDs):
```bash
python waste_classifier/scripts/draft.py
```
The model path is hardcoded to `/home/pi6/Desktop/IoT/waste_classifier/scripts/waste_classifier.tflite` — update if the Pi username or path differs.

**Run inference on Windows** (uses webcam, sends results to Flask):
```bash
cd waste_classifier/dataset
python use_model_Windows11.py
```

**Run the Flask dashboard server** (Windows, listens on port 8000):
```bash
cd waste_classifier/windows_flask
python HelloWorldTemplate.py
```
Dashboard at `http://localhost:8000/dashboard`. Web push notifications use VAPID keys auto-generated into `vapid_private.pem` / `vapid_public.txt` on first run.

## Architecture

### Model pipeline

`split_dataset.py` → `train_model.py` → `waste_classifier.tflite` → inference scripts

1. **Training** (`train_model.py`): MobileNetV2 (ImageNet weights, top removed) with a `GlobalAveragePooling2D → Dense(128, relu) → Dense(6, softmax)` head. Two-phase training: frozen backbone first, then fine-tune last 30 layers at `lr=1e-5`. Input: 224×224 RGB normalized to `[0,1]`.

2. **Inference**: The `.tflite` model is loaded via LiteRT (`ai_edge_litert`) on the Pi or `tf.lite.Interpreter` on Windows. Each frame is resized to 224×224, normalized, and the argmax of the 6-class softmax output is mapped to a bin.

### Waste categories (model output index → bin)

| Index | Bin label | Norwegian material |
|---|---|---|
| 0 | food_waste | matavfall |
| 1 | general_waste | restavfall |
| 2 | metal_glass | glass |
| 3 | metal_glass | metall |
| 4 | paper | papp og papir |
| 5 | plastic | plast |

Note: glass and metal map to the same physical bin (`metal_glass`). The dataset folders use separate `glass/` and `metal/` subdirectories; the mapping happens at inference time.

### Inference scripts

- `waste_classifier/dataset/use_model.py` — minimal Pi version (no LEDs, no confidence threshold)
- `waste_classifier/scripts/draft.py` — full Pi version with GPIO LEDs (`gpiozero`), confidence threshold (`UNDECIDED_THRESHOLD = 0.6`), throttled print/display/LED update intervals
- `waste_classifier/dataset/mellomlagring.py` — intermediate Pi version using `RPi.GPIO` directly instead of `gpiozero`
- `waste_classifier/dataset/use_model_Windows11.py` — Windows version with background subtraction (MOG2) to isolate foreground objects, bounding box overlay, and HTTP POST to Flask

### Flask server (`windows_flask/HelloWorldTemplate.py`)

Receives classification results from `use_model_Windows11.py` via `POST /api/classify`, stores the last 20 in memory, sends Web Push notifications to subscribed browsers, and serves a polling dashboard at `/dashboard`.

## Dependencies by environment

**Training (PC):** `tensorflow`, `keras`

**Raspberry Pi inference:** `ai-edge-litert`, `picamera2`, `opencv-python`, `numpy`, `RPi.GPIO` or `gpiozero`

**Windows inference:** `tensorflow`, `opencv-python`, `numpy`, `requests`

**Flask server:** `flask`, `pywebpush`, `cryptography`

## GPIO pin mapping (Raspberry Pi, BCM numbering)

| Bin | Pin | Component |
|---|---|---|
| metal_glass | 21 | Red LED |
| paper | 20 | Blue LED |
| plastic | 16 | RGB LED (green channel) |
| undecided | 17 | Single red LED |
