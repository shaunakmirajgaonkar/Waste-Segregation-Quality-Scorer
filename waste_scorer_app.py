"""
Waste-Segregation Quality Scorer
=================================
A 100% local, offline Python desktop application that analyzes an uploaded
image of waste (recyclables / compost / trash) and gives:
  - A visual "purity score" (how well-segregated the bin/image is)
  - Category-wise breakdown (Recyclable / Organic / Trash / Hazardous)
  - Contamination alerts
  - Actionable tips to improve segregation

No internet connection or external API is required. Image analysis uses
OpenCV + NumPy color/texture heuristics (a fast, dependency-light stand-in
for a trained CNN). If you later train a Keras/TensorFlow model, drop it in
as `waste_model.h5` and the app will automatically use it instead of the
heuristic engine (see `try_load_deep_model`).

Run:
    pip install pillow numpy opencv-python-headless
    python waste_scorer_app.py
"""

import os
import io
import json
import random
import threading
import time
from datetime import datetime

import numpy as np
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# --------------------------------------------------------------------------
# ---------------------------  COLOR THEME  --------------------------------
# --------------------------------------------------------------------------
THEME = {
    "bg":            "#0F172A",   # deep navy
    "panel":         "#1E293B",   # slate panel
    "panel_light":   "#273449",
    "accent":        "#22D3EE",   # cyan
    "accent2":       "#A78BFA",   # violet
    "good":          "#34D399",   # green
    "warn":          "#FBBF24",   # amber
    "bad":           "#F87171",   # red
    "text":          "#E2E8F0",
    "text_dim":      "#94A3B8",
    "recyclable":    "#38BDF8",
    "organic":       "#4ADE80",
    "trash":         "#FB923C",
    "hazardous":     "#F472B6",
}

CATEGORY_COLORS = {
    "Recyclable": THEME["recyclable"],
    "Organic":    THEME["organic"],
    "Trash":      THEME["trash"],
    "Hazardous":  THEME["hazardous"],
}

CATEGORY_TIPS = {
    "Recyclable": [
        "Rinse containers before placing them in the recycling bin.",
        "Flatten cardboard and plastic bottles to save space.",
        "Remove caps/lids if your local facility requires separation.",
        "Keep paper dry and free of food grease.",
    ],
    "Organic": [
        "Keep food scraps separate from plastic wrappers.",
        "Avoid putting compostable-labeled plastics in regular compost.",
        "Chop larger scraps to speed up composting.",
        "Layer green (wet) and brown (dry) organic waste for better compost.",
    ],
    "Trash": [
        "Double-check the item can't be recycled or composted first.",
        "Bag non-recyclable waste to prevent contamination of other bins.",
        "Reduce single-use trash items where possible.",
    ],
    "Hazardous": [
        "Never mix batteries, electronics, or chemicals with regular waste.",
        "Take hazardous items to a designated drop-off center.",
        "Store hazardous waste in sealed, labeled containers until disposal.",
    ],
}

HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scan_history.json")
DEEP_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "waste_model.h5")


# --------------------------------------------------------------------------
# ------------------------  ANALYSIS ENGINE  --------------------------------
# --------------------------------------------------------------------------
class WasteAnalysisEngine:
    """
    Fully local waste-image analysis engine.

    Primary mode: color / texture / shape heuristics (OpenCV + NumPy) that
    approximate material categories without needing any downloaded model
    or internet access.

    Optional deep-learning mode: if a Keras .h5 model file named
    'waste_model.h5' is present next to this script, it is loaded and used
    for classification instead. This lets you swap in a trained CNN later
    without changing the UI code.
    """

    LABELS = ["Recyclable", "Organic", "Trash", "Hazardous"]

    def __init__(self):
        self.deep_model = self.try_load_deep_model()

    # ---------------- optional deep model hook ----------------
    def try_load_deep_model(self):
        if not os.path.exists(DEEP_MODEL_PATH):
            return None
        try:
            import tensorflow as tf  # only imported if a model is actually present
            model = tf.keras.models.load_model(DEEP_MODEL_PATH)
            return model
        except Exception:
            return None

    # ---------------- main entry point ----------------
    def analyze(self, image_path: str) -> dict:
        bgr = cv2.imread(image_path)
        if bgr is None:
            # fall back to PIL for formats OpenCV struggles with, then convert
            pil_img = Image.open(image_path).convert("RGB")
            bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        bgr = self._resize_max(bgr, 640)

        if self.deep_model is not None:
            result = self._analyze_deep(bgr)
        else:
            result = self._analyze_heuristic(bgr)

        result["annotated_image"] = self._annotate(bgr, result)
        return result

    @staticmethod
    def _resize_max(img, max_dim=640):
        h, w = img.shape[:2]
        scale = max_dim / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        return img

    # ---------------- deep-learning branch (optional) ----------------
    def _analyze_deep(self, bgr):
        img = cv2.resize(bgr, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        preds = self.deep_model.predict(np.expand_dims(img, 0), verbose=0)[0]
        labels = self.LABELS[: len(preds)]
        scores = {lbl: float(p) for lbl, p in zip(labels, preds)}
        return self._package_result(scores, bgr)

    # ---------------- heuristic branch (default, 100% local) ----------------
    def _analyze_heuristic(self, bgr):
        """
        Segments the image into regions using color clustering, then scores
        each region against typical visual signatures of each waste class:

          Recyclable -> bright, saturated, uniform color blocks (plastic/
                         metal/glass sheen), strong edges (packaging shapes)
          Organic    -> browns/greens/dull earthy tones, high texture
                         irregularity (food scraps, peels, leaves)
          Trash      -> mixed/low-saturation grays, high entropy (mixed junk)
          Hazardous  -> small high-contrast regions with warning-like colors
                         (bright red/yellow/orange flags -> batteries, chemical
                         containers) - flagged conservatively as a caution
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # K-means color clustering to find dominant regions
        Z = bgr.reshape((-1, 3)).astype(np.float32)
        K = 5
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
        _, labels, centers = cv2.kmeans(Z, K, None, criteria, 4, cv2.KMEANS_RANDOM_CENTERS)
        labels = labels.flatten()
        counts = np.bincount(labels, minlength=K)
        total_px = counts.sum()

        edges = cv2.Canny(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 60, 160)
        edge_density = edges.sum() / 255.0 / total_px

        scores = {"Recyclable": 0.0, "Organic": 0.0, "Trash": 0.0, "Hazardous": 0.0}

        for i in range(K):
            frac = counts[i] / total_px
            b, g, r = centers[i]
            r, g, b = float(r), float(g), float(b)
            cluster_hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
            hue, sat, val = float(cluster_hsv[0]), float(cluster_hsv[1]), float(cluster_hsv[2])

            # Earthy brown/green -> organic
            is_brown = (5 <= hue <= 30) and sat > 40 and val < 190
            is_green = (35 <= hue <= 85) and sat > 40
            # Bright saturated plastics/metal/glass -> recyclable
            is_bright_sat = sat > 110 and val > 120
            # Warning colors (bright red/orange/yellow, small area) -> hazardous
            is_warning_color = ((hue < 12 or hue > 170) or (18 <= hue <= 30)) and sat > 150 and val > 140
            # Dull gray/low-sat -> trash
            is_dull = sat < 45

            if is_warning_color and frac < 0.25:
                scores["Hazardous"] += frac * 1.4
            elif is_brown or is_green:
                scores["Organic"] += frac * 1.3
            elif is_bright_sat:
                scores["Recyclable"] += frac * 1.3
            elif is_dull:
                scores["Trash"] += frac * 1.1
            else:
                # ambiguous cluster: split weight between trash/recyclable
                scores["Trash"] += frac * 0.5
                scores["Recyclable"] += frac * 0.3

        # Edge density boosts "Recyclable" (packaging has defined shapes)
        scores["Recyclable"] += min(edge_density * 2.0, 0.3)

        # Normalize to sum 1.0
        total = sum(scores.values()) or 1.0
        scores = {k: max(v / total, 0.0) for k, v in scores.items()}
        # renormalize after clipping
        total2 = sum(scores.values()) or 1.0
        scores = {k: v / total2 for k, v in scores.items()}

        return self._package_result(scores, bgr, edge_density=edge_density)

    def _package_result(self, scores, bgr, edge_density=None):
        dominant = max(scores, key=scores.get)
        sorted_scores = dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))

        # Purity score: how "confidently single-category" the image is.
        # High when one class dominates strongly (well-segregated bin);
        # low when categories are mixed evenly (contaminated / mixed waste).
        vals = sorted(scores.values(), reverse=True)
        purity = float(vals[0] - (vals[1] if len(vals) > 1 else 0))
        purity_pct = round(min(max(purity, 0), 1) * 60 + vals[0] * 40, 1)  # blend for nicer spread
        purity_pct = round(min(max(purity_pct, 0), 100), 1)

        contamination = []
        if sorted_scores.get("Hazardous", 0) > 0.12:
            contamination.append("Possible hazardous material detected (e.g., battery, chemical container). "
                                  "Remove immediately and dispose separately.")
        secondary = list(sorted_scores.items())[1] if len(sorted_scores) > 1 else None
        if secondary and secondary[1] > 0.28:
            contamination.append(f"Significant mixing with '{secondary[0]}' detected — separate before disposal.")
        if not contamination:
            contamination.append("No major contamination detected. Good segregation!")

        tips = list(CATEGORY_TIPS.get(dominant, []))
        random.shuffle(tips)
        tips = tips[:3]
        if secondary and secondary[1] > 0.2:
            extra = CATEGORY_TIPS.get(secondary[0], [])
            if extra:
                tips.append(random.choice(extra))

        grade = self._grade(purity_pct)

        return {
            "scores": sorted_scores,
            "dominant": dominant,
            "purity_pct": purity_pct,
            "grade": grade,
            "contamination": contamination,
            "tips": tips,
            "edge_density": edge_density,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    @staticmethod
    def _grade(pct):
        if pct >= 85:
            return ("A+", THEME["good"])
        elif pct >= 70:
            return ("A", THEME["good"])
        elif pct >= 55:
            return ("B", THEME["warn"])
        elif pct >= 40:
            return ("C", THEME["warn"])
        else:
            return ("D", THEME["bad"])

    @staticmethod
    def _annotate(bgr, result):
        """Draw a light contamination overlay / border color based on grade."""
        h, w = bgr.shape[:2]
        color_hex = result["grade"][1].lstrip("#")
        color_rgb = tuple(int(color_hex[i:i + 2], 16) for i in (0, 2, 4))
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        out = bgr.copy()
        thickness = 10
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), color_bgr, thickness)
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)


# --------------------------------------------------------------------------
# ---------------------------  UI COMPONENTS  --------------------------------
# --------------------------------------------------------------------------
class RoundedProgressBar(tk.Canvas):
    """A colorful custom progress/score bar."""

    def __init__(self, parent, width=280, height=22, **kwargs):
        super().__init__(parent, width=width, height=height,
                          bg=THEME["panel"], highlightthickness=0, **kwargs)
        self.width = width
        self.height = height
        self.set_value(0, THEME["accent"])

    def set_value(self, pct, color):
        self.delete("all")
        pct = max(0, min(100, pct))
        r = self.height / 2
        # track
        self._round_rect(2, 2, self.width - 2, self.height - 2, r, fill="#0B1220", outline="")
        # fill
        fill_w = max(self.height, (self.width - 4) * (pct / 100) + 4)
        self._round_rect(2, 2, fill_w, self.height - 2, r, fill=color, outline="")
        self.create_text(self.width / 2, self.height / 2, text=f"{pct:.1f}%",
                          fill="white", font=("Segoe UI", 10, "bold"))

    def _round_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
                  x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
        self.create_polygon(points, smooth=True, **kwargs)


class CategoryBar(tk.Frame):
    """Row showing category name + colored bar + percentage."""

    def __init__(self, parent, name, color, **kwargs):
        super().__init__(parent, bg=THEME["panel"], **kwargs)
        self.name = name
        self.color = color

        label = tk.Label(self, text=name, fg=color, bg=THEME["panel"],
                          font=("Segoe UI", 11, "bold"), width=12, anchor="w")
        label.pack(side="left", padx=(0, 8))

        self.bar = RoundedProgressBar(self, width=220, height=18)
        self.bar.pack(side="left", padx=4)

    def update_score(self, pct):
        self.bar.set_value(pct, self.color)


# --------------------------------------------------------------------------
# ------------------------------  MAIN APP  ---------------------------------
# --------------------------------------------------------------------------
class WasteScorerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("♻️  Waste-Segregation Quality Scorer")
        self.geometry("1150x760")
        self.minsize(980, 660)
        self.configure(bg=THEME["bg"])

        self.engine = WasteAnalysisEngine()
        self.current_image_path = None
        self.history = self._load_history()

        self._build_style()
        self._build_layout()

    # ---------------- ttk styling ----------------
    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TNotebook", background=THEME["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", background=THEME["panel"], foreground=THEME["text"],
                         padding=(16, 8), font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab", background=[("selected", THEME["accent"])],
                  foreground=[("selected", "#0B1220")])
        style.configure("Treeview", background=THEME["panel_light"], fieldbackground=THEME["panel_light"],
                         foreground=THEME["text"], rowheight=26, borderwidth=0, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", background=THEME["accent"], foreground="#0B1220",
                         font=("Segoe UI", 10, "bold"))
        style.map("Treeview", background=[("selected", THEME["accent2"])])

    # ---------------- layout ----------------
    def _build_layout(self):
        # ---- Header ----
        header = tk.Frame(self, bg=THEME["bg"])
        header.pack(fill="x", padx=24, pady=(18, 6))

        tk.Label(header, text="♻️  Waste-Segregation Quality Scorer", bg=THEME["bg"], fg=THEME["accent"],
                  font=("Segoe UI", 22, "bold")).pack(side="left")
        tk.Label(header, text="100% Local & Offline  •  No cloud, no data leaves your PC", bg=THEME["bg"],
                  fg=THEME["text_dim"], font=("Segoe UI", 10, "italic")).pack(side="left", padx=16, pady=(10, 0))

        # ---- Notebook (tabs) ----
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=20, pady=10)

        self.tab_scan = tk.Frame(self.nb, bg=THEME["bg"])
        self.tab_history = tk.Frame(self.nb, bg=THEME["bg"])
        self.tab_about = tk.Frame(self.nb, bg=THEME["bg"])

        self.nb.add(self.tab_scan, text="  🔍  Scan & Score  ")
        self.nb.add(self.tab_history, text="  📊  History Dashboard  ")
        self.nb.add(self.tab_about, text="  ℹ️  About  ")

        self._build_scan_tab()
        self._build_history_tab()
        self._build_about_tab()

    # ---------------- SCAN TAB ----------------
    def _build_scan_tab(self):
        container = tk.Frame(self.tab_scan, bg=THEME["bg"])
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)

        # LEFT: image panel
        left = tk.Frame(container, bg=THEME["panel"], bd=0)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=4)

        tk.Label(left, text="Upload Waste Image", bg=THEME["panel"], fg=THEME["text"],
                  font=("Segoe UI", 13, "bold")).pack(pady=(16, 8))

        self.image_canvas = tk.Label(left, bg="#0B1220", width=60, height=22,
                                      text="No image loaded\n\nClick 'Upload Image' below",
                                      fg=THEME["text_dim"], font=("Segoe UI", 11))
        self.image_canvas.pack(padx=20, pady=10, fill="both", expand=True)

        btn_row = tk.Frame(left, bg=THEME["panel"])
        btn_row.pack(pady=14)

        self._make_button(btn_row, "📁  Upload Image", self.on_upload, THEME["accent"]).pack(side="left", padx=6)
        self._make_button(btn_row, "🔎  Analyze", self.on_analyze, THEME["good"]).pack(side="left", padx=6)
        self._make_button(btn_row, "🗑️  Clear", self.on_clear, THEME["bad"]).pack(side="left", padx=6)

        # RIGHT: results panel
        right = tk.Frame(container, bg=THEME["panel"])
        right.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=4)

        tk.Label(right, text="Segregation Report", bg=THEME["panel"], fg=THEME["text"],
                  font=("Segoe UI", 13, "bold")).pack(pady=(16, 8))

        # Overall score gauge
        gauge_frame = tk.Frame(right, bg=THEME["panel"])
        gauge_frame.pack(pady=6)
        tk.Label(gauge_frame, text="Overall Purity Score", bg=THEME["panel"], fg=THEME["text_dim"],
                  font=("Segoe UI", 10)).pack()
        self.grade_label = tk.Label(gauge_frame, text="--", bg=THEME["panel"], fg=THEME["accent"],
                                     font=("Segoe UI", 42, "bold"))
        self.grade_label.pack()
        self.overall_bar = RoundedProgressBar(gauge_frame, width=320, height=24)
        self.overall_bar.pack(pady=6)

        # Category breakdown
        cat_frame = tk.LabelFrame(right, text=" Category Breakdown ", bg=THEME["panel"], fg=THEME["accent2"],
                                   font=("Segoe UI", 10, "bold"), bd=1, relief="groove")
        cat_frame.pack(fill="x", padx=20, pady=12)

        self.cat_bars = {}
        for name in ["Recyclable", "Organic", "Trash", "Hazardous"]:
            row = CategoryBar(cat_frame, name, CATEGORY_COLORS[name])
            row.pack(fill="x", padx=10, pady=4)
            self.cat_bars[name] = row

        # Contamination alerts
        alert_frame = tk.LabelFrame(right, text=" Contamination Alerts ", bg=THEME["panel"], fg=THEME["warn"],
                                     font=("Segoe UI", 10, "bold"), bd=1, relief="groove")
        alert_frame.pack(fill="x", padx=20, pady=6)
        self.alert_text = tk.Text(alert_frame, height=3, bg=THEME["panel_light"], fg=THEME["text"],
                                   font=("Segoe UI", 10), bd=0, wrap="word")
        self.alert_text.pack(fill="x", padx=6, pady=6)
        self.alert_text.insert("1.0", "Upload and analyze an image to see alerts here.")
        self.alert_text.config(state="disabled")

        # Tips
        tips_frame = tk.LabelFrame(right, text=" 💡 Improvement Tips ", bg=THEME["panel"], fg=THEME["good"],
                                    font=("Segoe UI", 10, "bold"), bd=1, relief="groove")
        tips_frame.pack(fill="both", expand=True, padx=20, pady=(6, 16))
        self.tips_text = tk.Text(tips_frame, bg=THEME["panel_light"], fg=THEME["text"],
                                  font=("Segoe UI", 10), bd=0, wrap="word")
        self.tips_text.pack(fill="both", expand=True, padx=6, pady=6)
        self.tips_text.config(state="disabled")

        self.status_var = tk.StringVar(value="Ready.")
        status_bar = tk.Label(self.tab_scan, textvariable=self.status_var, bg=THEME["bg"], fg=THEME["text_dim"],
                               font=("Segoe UI", 9), anchor="w")
        status_bar.pack(fill="x", padx=6, pady=(4, 0))

    def _make_button(self, parent, text, cmd, color):
        b = tk.Button(parent, text=text, command=cmd, bg=color, fg="#0B1220",
                      activebackground=color, activeforeground="#0B1220",
                      font=("Segoe UI", 10, "bold"), bd=0, padx=14, pady=8, cursor="hand2")
        return b

    # ---------------- HISTORY TAB ----------------
    def _build_history_tab(self):
        top = tk.Frame(self.tab_history, bg=THEME["bg"])
        top.pack(fill="x", padx=10, pady=10)
        tk.Label(top, text="Scan History & Trends", bg=THEME["bg"], fg=THEME["accent"],
                  font=("Segoe UI", 15, "bold")).pack(side="left")
        self._make_button(top, "🔄 Refresh", self.refresh_history, THEME["accent"]).pack(side="right", padx=4)
        self._make_button(top, "🧹 Clear History", self.clear_history, THEME["bad"]).pack(side="right", padx=4)

        # Stats cards
        stats_frame = tk.Frame(self.tab_history, bg=THEME["bg"])
        stats_frame.pack(fill="x", padx=10, pady=6)
        self.stat_cards = {}
        for i, key in enumerate(["Total Scans", "Avg Purity", "Best Grade", "Most Common"]):
            card = tk.Frame(stats_frame, bg=THEME["panel"], bd=0)
            card.grid(row=0, column=i, sticky="nsew", padx=6)
            stats_frame.columnconfigure(i, weight=1)
            tk.Label(card, text=key, bg=THEME["panel"], fg=THEME["text_dim"],
                      font=("Segoe UI", 9)).pack(pady=(10, 2))
            val = tk.Label(card, text="--", bg=THEME["panel"], fg=THEME["accent"], font=("Segoe UI", 18, "bold"))
            val.pack(pady=(0, 10))
            self.stat_cards[key] = val

        # Table
        table_frame = tk.Frame(self.tab_history, bg=THEME["bg"])
        table_frame.pack(fill="both", expand=True, padx=10, pady=10)

        cols = ("Timestamp", "File", "Dominant", "Purity %", "Grade")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=15)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=160, anchor="center")
        self.tree.pack(fill="both", expand=True, side="left")

        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        scroll.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=scroll.set)

        self.refresh_history()

    # ---------------- ABOUT TAB ----------------
    def _build_about_tab(self):
        frame = tk.Frame(self.tab_about, bg=THEME["bg"])
        frame.pack(fill="both", expand=True, padx=40, pady=30)

        tk.Label(frame, text="About This App", bg=THEME["bg"], fg=THEME["accent"],
                  font=("Segoe UI", 16, "bold")).pack(anchor="w", pady=(0, 12))

        about_text = (
            "Waste-Segregation Quality Scorer analyzes photos of waste/bins using "
            "local computer-vision techniques (OpenCV color clustering, edge density, "
            "and HSV heuristics) to estimate how well waste has been segregated into "
            "Recyclable, Organic, Trash, and Hazardous categories.\n\n"
            "✅ 100% Local — runs entirely on your machine, no internet or API key required.\n"
            "✅ Optional Deep Learning — drop a trained Keras model named 'waste_model.h5' "
            "next to this script to automatically switch to CNN-based classification.\n"
            "✅ History Dashboard — every scan is saved locally to scan_history.json so you "
            "can track your segregation quality over time.\n\n"
            "Tech stack: Python, Tkinter, OpenCV, NumPy, Pillow — all open-source, all local."
        )
        tk.Label(frame, text=about_text, bg=THEME["bg"], fg=THEME["text"], font=("Segoe UI", 11),
                  justify="left", wraplength=850).pack(anchor="w")

    # ----------------------------------------------------------------------
    # ------------------------------  LOGIC  --------------------------------
    # ----------------------------------------------------------------------
    def on_upload(self):
        path = filedialog.askopenfilename(
            title="Select waste image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")]
        )
        if not path:
            return
        self.current_image_path = path
        self._display_image(path)
        self.status_var.set(f"Loaded: {os.path.basename(path)}")

    def _display_image(self, path_or_pil):
        try:
            if isinstance(path_or_pil, str):
                img = Image.open(path_or_pil).convert("RGB")
            else:
                img = path_or_pil
            img.thumbnail((480, 420))
            tk_img = ImageTk.PhotoImage(img)
            self.image_canvas.configure(image=tk_img, text="")
            self.image_canvas.image = tk_img
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image:\n{e}")

    def on_clear(self):
        self.current_image_path = None
        self.image_canvas.configure(image="", text="No image loaded\n\nClick 'Upload Image' below")
        self.image_canvas.image = None
        self.grade_label.configure(text="--")
        self.overall_bar.set_value(0, THEME["accent"])
        for bar in self.cat_bars.values():
            bar.update_score(0)
        self._set_text(self.alert_text, "Upload and analyze an image to see alerts here.")
        self._set_text(self.tips_text, "")
        self.status_var.set("Cleared.")

    def on_analyze(self):
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return
        self.status_var.set("Analyzing... please wait.")
        self.update_idletasks()
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _run_analysis(self):
        try:
            result = self.engine.analyze(self.current_image_path)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Analysis Error", str(e)))
            self.after(0, lambda: self.status_var.set("Error during analysis."))
            return
        self.after(0, lambda: self._show_results(result))

    def _show_results(self, result):
        grade, color = result["grade"]
        self.grade_label.configure(text=grade, fg=color)
        self.overall_bar.set_value(result["purity_pct"], color)

        for name, bar in self.cat_bars.items():
            pct = result["scores"].get(name, 0) * 100
            bar.update_score(pct)

        self._set_text(self.alert_text, "\n".join(f"⚠️  {c}" if "hazard" in c.lower() or "mix" in c.lower()
                                                     else f"✅  {c}" for c in result["contamination"]))
        self._set_text(self.tips_text, "\n".join(f"• {t}" for t in result["tips"]))

        self._display_image(result["annotated_image"])
        self.status_var.set(f"Analysis complete — Dominant: {result['dominant']} "
                             f"({result['purity_pct']:.1f}% purity, Grade {grade})")

        self._save_history_entry(result)
        self.refresh_history()

    @staticmethod
    def _set_text(widget, text):
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.config(state="disabled")

    # ---------------- history persistence ----------------
    def _load_history(self):
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_history_entry(self, result):
        entry = {
            "timestamp": result["timestamp"],
            "file": os.path.basename(self.current_image_path) if self.current_image_path else "unknown",
            "dominant": result["dominant"],
            "purity_pct": result["purity_pct"],
            "grade": result["grade"][0],
            "scores": result["scores"],
        }
        self.history.append(entry)
        try:
            with open(HISTORY_FILE, "w") as f:
                json.dump(self.history, f, indent=2)
        except Exception:
            pass

    def refresh_history(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        for entry in reversed(self.history[-200:]):
            self.tree.insert("", "end", values=(
                entry["timestamp"], entry["file"], entry["dominant"],
                f"{entry['purity_pct']:.1f}", entry["grade"]
            ))

        total = len(self.history)
        avg_purity = round(sum(e["purity_pct"] for e in self.history) / total, 1) if total else 0
        best_grade = min((e["grade"] for e in self.history), default="--",
                          key=lambda g: {"A+": 0, "A": 1, "B": 2, "C": 3, "D": 4}.get(g, 9))
        if total:
            from collections import Counter
            most_common = Counter(e["dominant"] for e in self.history).most_common(1)[0][0]
        else:
            most_common = "--"

        self.stat_cards["Total Scans"].configure(text=str(total))
        self.stat_cards["Avg Purity"].configure(text=f"{avg_purity}%")
        self.stat_cards["Best Grade"].configure(text=best_grade)
        self.stat_cards["Most Common"].configure(text=most_common)

    def clear_history(self):
        if not self.history:
            return
        if messagebox.askyesno("Confirm", "Clear all scan history? This cannot be undone."):
            self.history = []
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            self.refresh_history()


# --------------------------------------------------------------------------
if __name__ == "__main__":
    app = WasteScorerApp()
    app.mainloop()
