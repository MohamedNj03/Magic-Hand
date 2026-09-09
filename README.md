# Magic Hand

**Draw, write and manipulate 3D objects in the air — with nothing but a webcam.**

No controller, no touchscreen, no depth sensor, no cloud API.


---

## Gestures

| Gesture | Action |
|---|---|
| ✏️ **Index finger, held** | Draw. Pause briefly and the stroke becomes a 3D object |
| ✊ **Closed fist** | Grab and move it — anything stacked on top comes along |
| 🖐 **Open hand over it** | Rotate it in 3D, in place |
| 👉 **Move your hand away** | Deselect |
| 🤏 **Circle sign, 3 fingers** | Magnifying glass. With an object selected, **raise** the three fingers to grow it, **lower** them to shrink it |
| 👍 **Thumb up** | Next colour |
| 👎 **Thumb down** | Delete the last drawing |
| 🙌 **Both hands open, then removed from frame** | Clear the canvas |

**Keys** — `q` quit · `h` help · `o` shadows · `c` colour · `z` undo · `s` save a PNG

One hand drives everything. The second hand exists only to arm the canvas wipe — the single destructive action, and the only one a hand busy drawing cannot trigger by accident.

**Clearing the canvas** happens in two stages: raise both open palms until the screen reads *"NOW REMOVE BOTH HANDS"*, then take both hands out of frame. Showing a hand again cancels it, and `z` restores everything afterwards.

---

## Installation

```bash
git clone https://github.com/MohamedNj03/Magic-Hand.git
cd magic-hand

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

### Download the hand model

**Windows — PowerShell**

```powershell
mkdir models
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task" -OutFile "models\gesture_recognizer.task"
```

> **Why `Invoke-WebRequest` and not `curl` here.** In PowerShell, `curl` is an
> alias for `Invoke-WebRequest`, which understands neither `-L` nor `-o` (it
> expects `-OutFile`), and `\` is not a line continuation in PowerShell. A
> `curl -L -o ...` command fails before it ever reaches the network. To use the
> real curl on Windows, call it by its full name: `curl.exe`.

<details>
<summary>macOS / Linux</summary>

```bash
mkdir models
curl -L -o models/gesture_recognizer.task \
  https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
```
</details>

This 8 MB file is Google's pre-trained MediaPipe model: it turns each camera frame into 21 hand landmarks. It is not committed here because it is a third-party binary, and Git handles large binaries poorly. The app loads it from `models/gesture_recognizer.task` and reports an explicit error if it is missing.

### Run

```bash
python gesture_canvas_manipulation.py
```

Handwriting recognition uses Tesseract when the binary is installed on your system, and falls back to a built-in template matcher when it is not — the app works either way. Tesseract does not need to be on your `PATH`; it is looked up at the usual install locations.

---

## Folders not in this repository

| Folder | What it is |
|---|---|
| `venv/` | Your virtual environment — machine-specific, hundreds of MB |
| `models/` | The MediaPipe model, downloaded with the command above |
| `__pycache__/` | Python bytecode cache, regenerated on every run |
| `captures/` | Screenshots saved with the `s` key |

Each is machine-specific, regenerated automatically, or fetched from its official source. The repository holds only hand-written source.

---

## How it works

MediaPipe provides 21 hand landmarks per frame. Poses are then read **geometrically** rather than by trusting a classifier, and the result flows through eight stages:

```
camera frame
   │
   ├─ 1  hand_signals.py        read the hand: poses, orientation, lens circle
   ├─ 2  filters.py             One Euro smoothing on every noisy signal
   ├─ 3  gesture_canvas_...py   decide what the pose MEANS (state machines)
   │        └─ canvas.py        strokes, objects, stacking, undo history
   ├─ 4  sketch_recognition.py  a finished drawing -> shape, letter, or left alone
   ├─ 5  async_recognition.py   the slow half, on a worker thread
   ├─ 6  mesh3d.py              rotate, project, depth-sort, shade, cast a shadow
   ├─ 7  magnifier.py           the zoom lens
   └─ 8  hud.py                 the overlay
```

Each file opens with a header stating its stage and what it connects to.

---

## Project structure

| File | Role |
|---|---|
| `gesture_canvas_manipulation.py` | Entry point: gestures, state machines, camera loop |
| `hand_signals.py` | Geometric reading of the hand from MediaPipe landmarks |
| `filters.py` | One Euro smoothing, dead zone, angle helpers |
| `canvas.py` | Strokes, 3D objects, stacking, rendering, undo |
| `sketch_recognition.py` | Stroke cleanup, shape detection, handwriting |
| `async_recognition.py` | Recognition worker thread |
| `mesh3d.py` | 3D engine: rotation, projection, shading, shadows |
| `magnifier.py` | The zoom lens |
| `hud.py` | On-screen overlay |

---

## Limitations

- **No true depth.** A single RGB camera cannot measure distance: yaw and pitch are noisy relative proxies. What is genuinely 3D is the object — real faces, real edges, real occlusion.
- **Six shapes only** — circle, triangle, square, rectangle, pentagon, hexagon. Anything else stays a drawing on purpose. A hexagon drawn quickly may still read as a circle: once the corners are rounded off, the two are genuinely indistinguishable.
- **A genuinely round scribble reads as a circle.** It scores better on every roundness measure than a real hand-drawn circle, so no threshold separates them without rejecting honest circles too.
- **Incomplete shapes can sometimes be misclassified.** When a shape is not drawn completely or some of its edges are missing, the system may occasionally interpret the partial drawing as another supported shape. This is an inherent limitation of recognizing shapes from incomplete contours.
- **Offline letter recognition** (no Tesseract) reaches ~90 % on capitals and digits. Failures are abstentions, never inventions.
- **A closed outline must be read very clearly to count as a letter.** It is far more likely to be a shape this file declined to name, so the bar is raised and the offline guesser is skipped entirely. A letter written on purpose still clears it; a failed triangle no longer comes back as a "V".
- **Sensitivities may need tuning** to your camera and lighting: `YAW_SENSITIVITY`, `PITCH_SENSITIVITY` and `DEPTH_DEADZONE` in `gesture_canvas_manipulation.py`.

---

## Built with

Python · OpenCV · MediaPipe · NumPy · Tesseract (optional)

---

## License

MIT — see [LICENSE](LICENSE).

Copyright (c) 2026 MohamedNj03

The MediaPipe hand model downloaded during setup is Google's, under its own
licence, and is not distributed with this repository.
