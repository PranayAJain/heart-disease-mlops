"""Single source of truth for paths and column metadata."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = ROOT / "data" / "raw" / "processed.cleveland.data"
CLEAN_FILE = ROOT / "data" / "processed" / "heart_clean.csv"
FIG_DIR = ROOT / "reports" / "figures"

# Column order of the raw UCI file (the file has no header)
RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
TARGET = "target"  # binary: 1 = heart disease present (num > 0)

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY = ["sex", "fbs", "exang"]
CATEGORICAL = ["cp", "restecg", "slope", "thal"]
FEATURES = NUMERIC + BINARY + CATEGORICAL

# Human-readable codes (from the UCI documentation) - used in EDA and API docs
CODE_LABELS = {
    "sex": {0: "female", 1: "male"},
    "cp": {1: "typical angina", 2: "atypical angina", 3: "non-anginal", 4: "asymptomatic"},
    "fbs": {0: "<=120 mg/dl", 1: ">120 mg/dl"},
    "restecg": {0: "normal", 1: "ST-T abnormality", 2: "LV hypertrophy"},
    "exang": {0: "no", 1: "yes"},
    "slope": {1: "upsloping", 2: "flat", 3: "downsloping"},
    "thal": {3: "normal", 6: "fixed defect", 7: "reversible defect"},
}

# Plausible physiological ranges - used in cleaning and API validation
VALID_RANGES = {
    "age": (18, 100),
    "trestbps": (80, 220),
    "chol": (100, 600),
    "thalach": (60, 220),
    "oldpeak": (0.0, 7.0),
    "ca": (0, 3),
}
