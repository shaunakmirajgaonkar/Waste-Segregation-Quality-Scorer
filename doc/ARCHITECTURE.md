# Architecture

## Processing Pipeline

Waste Image → Image Loading → Preprocessing → HSV Conversion → K-Means Clustering → Visual Analysis → Category Scores → Purity Score → Grade → Contamination Detection → Recommendations → Local History

## Main Components

### WasteAnalysisEngine

Handles image loading, preprocessing, computer-vision analysis, category scoring, purity calculation, contamination detection, and optional CNN inference.

### WasteScorerApp

Handles the Tkinter interface, image upload, analysis, results, history dashboard, and recommendations.

### History

Results are stored locally in scan_history.json.
