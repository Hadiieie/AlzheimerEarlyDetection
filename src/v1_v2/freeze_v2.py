from pathlib import Path
import hashlib
import json
import shutil
import sys

import joblib
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline


# ============================================================
# FREEZE V2
# Alzheimer MRI Early Detection
#
# PURPOSE:
# Freeze the already-selected V2 candidate exactly as-is.
#
# LOCKED V2:
#   SVM RBF
#   C = 50
#   gamma = 0.06
#   class_weight = balanced
#   threshold = selected on validation only
#
# IMPORTANT:
# - DOES NOT retrain the model.
# - DOES NOT load validation.csv.
# - DOES NOT load test.csv.
# - DOES NOT alter V1.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

SOURCE_DIR = RESULTS_DIR / "V2_VALIDATION_THRESHOLD"
V1_DIR = RESULTS_DIR / "FINAL_MODEL"
FINAL_V2_DIR = RESULTS_DIR / "FINAL_MODEL_V2"

FINAL_V2_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_MODEL = SOURCE_DIR / "v2_candidate_model_pre_freeze.joblib"
SOURCE_THRESHOLD = SOURCE_DIR / "v2_selected_threshold.json"
FEATURE_FILE = V1_DIR / "final_features.txt"

FINAL_MODEL = FINAL_V2_DIR / "final_v2_model.joblib"
FINAL_SCALER = FINAL_V2_DIR / "final_v2_scaler.joblib"
FINAL_FEATURES = FINAL_V2_DIR / "final_v2_features.txt"
FINAL_CONFIG = FINAL_V2_DIR / "final_v2_model_config.json"
FREEZE_MANIFEST = FINAL_V2_DIR / "freeze_manifest_v2.json"
CHECKSUMS_FILE = FINAL_V2_DIR / "SHA256_CHECKSUMS_V2.txt"


EXPECTED_C = 50.0
EXPECTED_GAMMA = 0.06
EXPECTED_KERNEL = "rbf"
EXPECTED_CLASS_WEIGHT = "balanced"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_features(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


print()
print("=" * 88)
print("FREEZE V2")
print("=" * 88)
print()
print("DATA POLICY")
print("-----------")
print("No train/validation/test CSV is loaded by this script.")
print("The selected V2 candidate is frozen exactly as already trained.")
print()

for path in [SOURCE_MODEL, SOURCE_THRESHOLD, FEATURE_FILE]:
    if not path.exists():
        raise FileNotFoundError(path)

features = load_features(FEATURE_FILE)

if len(features) != 30:
    raise ValueError(
        f"Expected 30 frozen features, found {len(features)}."
    )

with open(SOURCE_THRESHOLD, "r", encoding="utf-8") as f:
    threshold_info = json.load(f)

selected_threshold = float(
    threshold_info["selected_threshold"]
)

locked_info = threshold_info.get("locked_model", {})

if not np.isclose(
    float(locked_info.get("C")),
    EXPECTED_C,
    atol=1e-15,
    rtol=0.0,
):
    raise RuntimeError("Threshold artifact C does not match expected V2.")

if not np.isclose(
    float(locked_info.get("gamma")),
    EXPECTED_GAMMA,
    atol=1e-15,
    rtol=0.0,
):
    raise RuntimeError(
        "Threshold artifact gamma does not match expected V2."
    )

if locked_info.get("kernel") != EXPECTED_KERNEL:
    raise RuntimeError("Threshold artifact kernel mismatch.")

if locked_info.get("class_weight") != EXPECTED_CLASS_WEIGHT:
    raise RuntimeError("Threshold artifact class_weight mismatch.")

if bool(threshold_info.get("test_set_loaded", True)):
    raise RuntimeError(
        "Threshold artifact indicates the test set may have been loaded."
    )

model = joblib.load(SOURCE_MODEL)

if not isinstance(model, Pipeline):
    raise TypeError("V2 candidate model is not a sklearn Pipeline.")

if list(model.named_steps.keys()) != ["scaler", "clf"]:
    raise ValueError(
        "Expected pipeline steps exactly ['scaler', 'clf']."
    )

clf = model.named_steps["clf"]
params = clf.get_params()

checks = {
    "kernel": params.get("kernel") == EXPECTED_KERNEL,
    "C": np.isclose(
        float(params.get("C")),
        EXPECTED_C,
        atol=1e-15,
        rtol=0.0,
    ),
    "gamma": np.isclose(
        float(params.get("gamma")),
        EXPECTED_GAMMA,
        atol=1e-15,
        rtol=0.0,
    ),
    "class_weight": (
        params.get("class_weight") == EXPECTED_CLASS_WEIGHT
    ),
}

if not all(checks.values()):
    raise RuntimeError(
        f"Candidate model parameter verification failed: {checks}"
    )

print("Candidate model parameter verification: PASS")
print(f"C            = {params['C']}")
print(f"gamma        = {params['gamma']}")
print(f"kernel       = {params['kernel']}")
print(f"class_weight = {params['class_weight']}")
print(f"threshold    = {selected_threshold:.15f}")
print(f"features     = {len(features)}")

# ------------------------------------------------------------
# Save exact frozen artifacts.
# We intentionally re-dump the loaded pipeline as FINAL V2.
# No refitting occurs.
# ------------------------------------------------------------

joblib.dump(model, FINAL_MODEL)
joblib.dump(model.named_steps["scaler"], FINAL_SCALER)

shutil.copyfile(FEATURE_FILE, FINAL_FEATURES)

config = {
    "version": "V2",
    "status": "FROZEN_BEFORE_FINAL_TEST",
    "model_type": "sklearn Pipeline(StandardScaler -> SVC RBF)",
    "model_parameters": {
        "kernel": EXPECTED_KERNEL,
        "C": EXPECTED_C,
        "gamma": EXPECTED_GAMMA,
        "class_weight": EXPECTED_CLASS_WEIGHT,
    },
    "selected_threshold": selected_threshold,
    "threshold_source": (
        "Validation-only exhaustive threshold selection"
    ),
    "hyperparameter_source": (
        "Training-only 3-fold StratifiedGroupKFold CV "
        "using leakage_group"
    ),
    "feature_count": len(features),
    "features": features,
    "test_status": "NOT_EVALUATED_AT_FREEZE_TIME",
    "important": (
        "Do not change model hyperparameters, feature order, scaler, "
        "or threshold after this freeze. The held-out test set must be "
        "evaluated once with this exact frozen package."
    ),
}

with open(FINAL_CONFIG, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4)

# Compute checksums only after all artifacts exist.
artifact_paths = [
    FINAL_MODEL,
    FINAL_SCALER,
    FINAL_FEATURES,
    FINAL_CONFIG,
]

checksums = {
    path.name: sha256_file(path)
    for path in artifact_paths
}

manifest = {
    "version": "V2",
    "status": "FROZEN_BEFORE_FINAL_TEST",
    "python_version": sys.version,
    "scikit_learn_version": sklearn.__version__,
    "selected_threshold": selected_threshold,
    "model_parameters": {
        "kernel": EXPECTED_KERNEL,
        "C": EXPECTED_C,
        "gamma": EXPECTED_GAMMA,
        "class_weight": EXPECTED_CLASS_WEIGHT,
    },
    "feature_count": len(features),
    "source_candidate_model": str(SOURCE_MODEL),
    "source_threshold_artifact": str(SOURCE_THRESHOLD),
    "artifacts": {
        "model": FINAL_MODEL.name,
        "scaler": FINAL_SCALER.name,
        "features": FINAL_FEATURES.name,
        "config": FINAL_CONFIG.name,
    },
    "sha256": checksums,
    "test_set_evaluated": False,
}

with open(FREEZE_MANIFEST, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=4)

# Add manifest checksum after writing it.
checksums[FREEZE_MANIFEST.name] = sha256_file(FREEZE_MANIFEST)

with open(CHECKSUMS_FILE, "w", encoding="utf-8") as f:
    for name, digest in checksums.items():
        f.write(f"{digest}  {name}\n")

print()
print("=" * 88)
print("V2 FREEZE COMPLETE")
print("=" * 88)
print()
print("FINAL V2 IS NOW LOCKED.")
print("The test set has NOT been loaded or evaluated.")
print()
print("Frozen model SHA256:")
print(checksums[FINAL_MODEL.name])
print()
print("Saved:")
print(FINAL_MODEL)
print(FINAL_SCALER)
print(FINAL_FEATURES)
print(FINAL_CONFIG)
print(FREEZE_MANIFEST)
print(CHECKSUMS_FILE)
print()
print("Next step: one-time held-out V2 test evaluation.")
