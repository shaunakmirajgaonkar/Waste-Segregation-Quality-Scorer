# ♻️ Waste-Segregation Quality Scorer

A **100% local, offline** Python desktop app that analyzes a photo of waste
and scores how well it has been segregated (Recyclable / Organic / Trash /
Hazardous), flags contamination, and gives improvement tips — all with a
colorful, professional Tkinter dashboard.

## Features
- 📁 Upload any image (jpg/png/bmp/webp)
- 🔎 One-click local analysis (OpenCV color-clustering + edge/texture heuristics — no internet, no API keys)
- 🎨 Colorful animated-style progress bars per category
- ⚠️ Automatic contamination alerts (e.g. hazardous items mixed in)
- 💡 Context-aware improvement tips
- 📊 History dashboard with stats (total scans, average purity, best grade, most common category) saved locally to `scan_history.json`
- 🧠 Optional deep-learning mode: drop a trained Keras model as `waste_model.h5` next to the script and it's used automatically instead of the heuristic engine

## Setup
```bash
pip install -r requirements.txt
python waste_scorer_app.py
```

Requires Python 3.8+. Works on Windows / macOS / Linux.

## How the scoring works
1. Image is loaded and resized locally with OpenCV.
2. K-means color clustering finds the dominant color regions.
3. Each region's HSV signature is matched against typical material
   signatures (bright saturated = recyclable plastics/metal/glass; earthy
   brown/green = organic; low-saturation gray = general trash; small
   high-contrast warning colors = potential hazardous items).
4. Edge density (from Canny edge detection) adds weight toward
   "Recyclable" since packaging tends to have well-defined shapes.
5. Scores are normalized into percentages; the top category is the
   "dominant" class, and the gap between the top two scores drives the
   overall **Purity Score** and letter grade (A+ to D).
6. Every scan is appended to `scan_history.json` for the dashboard tab.

## Upgrading to a real trained CNN (optional)
If you train a Keras/TensorFlow classifier (e.g., on the TrashNet or
similar waste dataset) with output classes in the order
`["Recyclable", "Organic", "Trash", "Hazardous"]`, save it as:
```
waste_model.h5
```
in the same folder as `waste_scorer_app.py`. The app detects the file on
startup and automatically routes all analysis through your model instead
of the built-in heuristic engine — no code changes needed.

## Files
- `waste_scorer_app.py` — the full application (UI + analysis engine)
- `requirements.txt` — Python dependencies
- `scan_history.json` — auto-created after your first scan
