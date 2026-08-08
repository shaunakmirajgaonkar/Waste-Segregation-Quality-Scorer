# API / Programmatic Interface

The current project is a desktop application and does not expose a REST API.

## Programmatic Usage

    from waste_scorer_app import WasteAnalysisEngine
    engine = WasteAnalysisEngine()
    result = engine.analyze("image.jpg")

## Result Fields

- scores
- dominant
- purity_pct
- grade
- contamination
- tips
- edge_density
- timestamp
- annotated_image
