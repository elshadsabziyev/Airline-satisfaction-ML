# Airline Satisfaction Lab

Interactive workspace that trains several classical ML models (logistic, linear & polynomial regression, KNN, decision tree, calibrated linear SVM) on the airline satisfaction survey, then serves a Streamlit experience for mobile/desktop. The web app guides a passenger-style intake, shows personalized drivers vs. the empirical base rate, and surfaces full diagnostics for every model.

## Project layout

```
├── data/                 # train/test csv files (unchanged)
├── artifacts/
│   ├── models/           # serialized sklearn pipelines (.joblib)
│   └── model_registry.json  # metrics, CV stats, feature ranks
├── src/
│   ├── models/
│   │   └── train_models.py  # training + registry generation script
│   └── app.py             # Streamlit UI
├── requirements.txt
└── README.md
```

## Environment setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train models (mandatory before running the app)

```powershell
.venv\Scripts\python.exe src/models/train_models.py
```

This script

- cleans the survey data consistently,
- fits each requested model with per-fold cross-validation,
- logs train/test metrics, confusion matrices, coefficient-level feature signals, and
- writes everything into `artifacts/model_registry.json` + `.joblib` artifacts for inference.

## Run the Streamlit assistant

```powershell
streamlit run src/app.py
```

Highlights:

- **Passenger wizard** – multi-step form with progress, popovers, and sliders optimized for mobile touch targets.
- **Model lab** – select any model to view accuracy/precision/recall/F1, confusion matrices, CV spread, and side-by-side comparisons.
- **Global insights** – cross-model feature ranking, base-rate context, and dataset notes for scientific traceability.
- **Personal narrative** – per-response driver bars built from logistic coefficients plus actionable airline recommendations.

## Repro checklist

1. Confirm `data/train.csv` and `data/test.csv` exist (see `dataset.md` for schema).
2. Install dependencies (`pip install -r requirements.txt`).
3. Run `src/models/train_models.py` to refresh artifacts whenever data or code changes.
4. Launch `streamlit run src/app.py` and explore the tabs.
5. Optional: version artifacts for experiment tracking.

## Notes

- Streamlit 1.51 components (segmented controls, popovers, progress indicators) were used to keep feature parity across desktop & mobile.
- SVM training now uses a calibrated linear SVM to avoid the hang you observed with RBF kernels while still exposing probabilities.
- Feature attributions remain tethered to actual logistic coefficients to avoid leakage and to maximize scientific transparency.
