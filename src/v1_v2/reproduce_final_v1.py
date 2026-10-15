from pathlib import Path
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# V1 EXACT REPRODUCTION
# Alzheimer MRI Early Detection
#
# PURPOSE:
# Refit the already-frozen V1 configuration from scratch using
# the original TRAIN split, then compare it with the frozen V1.
#
# THIS SCRIPT DOES NOT:
# - tune hyperparameters
# - change features
# - optimize threshold
# - overwrite the frozen model
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
SPLIT_DIR = RESULTS_DIR / "GROUP_AWARE_SPLIT"
FINAL_DIR = RESULTS_DIR / "FINAL_MODEL"
OUTPUT_DIR = RESULTS_DIR / "V1_REPRODUCIBILITY"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = SPLIT_DIR / "train.csv"
VAL_FILE = SPLIT_DIR / "validation.csv"
TEST_FILE = SPLIT_DIR / "test.csv"

FEATURE_FILE = FINAL_DIR / "final_features.txt"
CONFIG_FILE = FINAL_DIR / "final_model_config.json"
FROZEN_MODEL_FILE = FINAL_DIR / "final_model.joblib"
FROZEN_SCALER_FILE = FINAL_DIR / "final_scaler.joblib"

REPRO_MODEL_FILE = OUTPUT_DIR / "reproduced_v1_model.joblib"
REPORT_JSON = OUTPUT_DIR / "v1_reproducibility_report.json"
REPORT_TXT = OUTPUT_DIR / "v1_reproducibility_report.txt"


# ============================================================
# 2. FROZEN V1 CONFIGURATION
# ============================================================

EXPECTED_SKLEARN_VERSION = "1.9.1"

EXPECTED_ROWS = {
    "train": 20891,
    "validation": 4482,
    "test": 4480,
}

EXPECTED_PARAMS = {
    "kernel": "rbf",
    "C": 25,
    "gamma": 0.05,
    "class_weight": "balanced",
}

FROZEN_THRESHOLD = 0.10872472921801979

EXPECTED_VALIDATION = {
    "accuracy": 0.8130298973672467,
    "balanced_accuracy": 0.8114150074795645,
    "sensitivity": 0.8162923231645994,
    "specificity": 0.8065376917945297,
    "precision": 0.8935779816513761,
    "f1": 0.8531885073580939,
    "roc_auc": 0.8882752766007599,
    "pr_auc": 0.9291188728458959,
    "TN": 1209,
    "FP": 290,
    "FN": 548,
    "TP": 2435,
}

EXPECTED_TEST = {
    "accuracy": 0.812054,
    "balanced_accuracy": 0.810185,
    "sensitivity": 0.815834,
    "specificity": 0.804536,
    "precision": 0.892477,
    "f1": 0.852436,
    "roc_auc": 0.887921,
    "pr_auc": 0.929306,
    "TN": 1206,
    "FP": 293,
    "FN": 549,
    "TP": 2432,
}


# ============================================================
# 3. HELPERS
# ============================================================

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


def validate_dataframe(name, df, features):
    required = {"target", "leakage_group"}
    missing_required = sorted(required - set(df.columns))
    if missing_required:
        raise ValueError(
            f"{name}: missing required columns: {missing_required}"
        )

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(
            f"{name}: missing feature columns: {missing_features}"
        )

    if len(df) != EXPECTED_ROWS[name]:
        raise ValueError(
            f"{name}: expected {EXPECTED_ROWS[name]} rows, found {len(df)}"
        )

    targets = set(pd.unique(df["target"]))
    if not targets.issubset({0, 1}):
        raise ValueError(
            f"{name}: target must contain only 0/1, found {targets}"
        )

    X = df[features].to_numpy(dtype=float)

    nan_count = int(np.isnan(X).sum())
    inf_count = int(np.isinf(X).sum())

    if nan_count != 0 or inf_count != 0:
        raise ValueError(
            f"{name}: invalid feature values: "
            f"NaN={nan_count}, Inf={inf_count}"
        )


def metrics_from_scores(y_true, scores, threshold):
    y_pred = (scores >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "sensitivity": float(
            recall_score(y_true, y_pred, pos_label=1)
        ),
        "specificity": float(specificity),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "f1": float(f1_score(y_true, y_pred, pos_label=1)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def compare_metrics(actual, expected, rounded_expected=False):
    result = {}
    all_pass = True

    for key, expected_value in expected.items():
        actual_value = actual[key]

        if key in {"TN", "FP", "FN", "TP"}:
            passed = int(actual_value) == int(expected_value)
            difference = int(actual_value) - int(expected_value)
        else:
            if rounded_expected:
                # Saved final test metrics are stored to 6 decimals.
                passed = round(float(actual_value), 6) == round(
                    float(expected_value), 6
                )
            else:
                passed = np.isclose(
                    float(actual_value),
                    float(expected_value),
                    rtol=0.0,
                    atol=1e-12,
                )

            difference = float(actual_value) - float(expected_value)

        result[key] = {
            "actual": actual_value,
            "expected": expected_value,
            "difference": difference,
            "pass": bool(passed),
        }

        if not passed:
            all_pass = False

    return all_pass, result


# ============================================================
# 4. HEADER
# ============================================================

print()
print("=" * 78)
print("V1 EXACT REPRODUCTION")
print("=" * 78)
print()
print("Project root:")
print(PROJECT_ROOT)

print("\nscikit-learn version:")
print(sklearn.__version__)

if sklearn.__version__ != EXPECTED_SKLEARN_VERSION:
    raise RuntimeError(
        "\nExact V1 reproduction requires scikit-learn "
        f"{EXPECTED_SKLEARN_VERSION}, but this interpreter has "
        f"{sklearn.__version__}.\n"
        "Use the same Python environment that produced the frozen model."
    )


# ============================================================
# 5. VERIFY INPUT FILES
# ============================================================

required_files = [
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
    FEATURE_FILE,
    CONFIG_FILE,
    FROZEN_MODEL_FILE,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )

print("\nRequired files: PASS")


# ============================================================
# 6. LOAD FROZEN CONFIG + FEATURES
# ============================================================

features = load_features(FEATURE_FILE)

if len(features) != 30:
    raise ValueError(
        f"Expected exactly 30 frozen features, found {len(features)}"
    )

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    frozen_config = json.load(f)

config_threshold = float(frozen_config["selected_threshold"])

if not np.isclose(
    config_threshold,
    FROZEN_THRESHOLD,
    rtol=0.0,
    atol=1e-15,
):
    raise ValueError(
        "Frozen threshold in final_model_config.json does not match "
        "the expected V1 threshold."
    )

config_features = frozen_config.get("features", [])

if config_features != features:
    raise ValueError(
        "Feature order in final_model_config.json does not exactly "
        "match final_features.txt."
    )

print("Frozen feature list/config: PASS")


# ============================================================
# 7. LOAD ORIGINAL RAW SPLITS
# ============================================================

print("\nLoading original group-aware splits...")

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)

validate_dataframe("train", train_df, features)
validate_dataframe("validation", val_df, features)
validate_dataframe("test", test_df, features)

print("Split shapes / numerical QC: PASS")


# ============================================================
# 8. GROUP OVERLAP CHECK
# ============================================================

train_groups = set(train_df["leakage_group"].astype(str))
val_groups = set(val_df["leakage_group"].astype(str))
test_groups = set(test_df["leakage_group"].astype(str))

overlaps = {
    "train_validation": len(train_groups & val_groups),
    "train_test": len(train_groups & test_groups),
    "validation_test": len(val_groups & test_groups),
}

print("\nGroup overlap:")
for key, value in overlaps.items():
    print(f"{key}: {value}")

if any(value != 0 for value in overlaps.values()):
    raise RuntimeError(
        "Leakage-group overlap detected. Reproduction aborted."
    )

print("Group leakage QC: PASS")


# ============================================================
# 9. MATRICES
# ============================================================

X_train = train_df[features]
y_train = train_df["target"].astype(int).to_numpy()

X_val = val_df[features]
y_val = val_df["target"].astype(int).to_numpy()

X_test = test_df[features]
y_test = test_df["target"].astype(int).to_numpy()


# ============================================================
# 10. REBUILD EXACT FROZEN MODEL CONFIGURATION
# ============================================================

print()
print("=" * 78)
print("REFITTING FROZEN V1 CONFIGURATION")
print("=" * 78)

reproduced_model = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "clf",
            SVC(
                kernel="rbf",
                C=25,
                gamma=0.05,
                class_weight="balanced",
                probability=False,
                shrinking=True,
                tol=0.001,
                cache_size=200,
                max_iter=-1,
                decision_function_shape="ovr",
                break_ties=False,
            ),
        ),
    ]
)

reproduced_model.fit(X_train, y_train)

print("Training complete.")


# ============================================================
# 11. LOAD FROZEN MODEL
# ============================================================

frozen_model = joblib.load(FROZEN_MODEL_FILE)

if not isinstance(frozen_model, Pipeline):
    raise TypeError("Frozen model is not a sklearn Pipeline.")

if list(frozen_model.named_steps.keys()) != ["scaler", "clf"]:
    raise ValueError(
        "Frozen pipeline steps are not exactly ['scaler', 'clf']."
    )

frozen_params = frozen_model.named_steps["clf"].get_params()

for key, expected in EXPECTED_PARAMS.items():
    actual = frozen_params[key]
    if actual != expected:
        raise ValueError(
            f"Frozen model parameter mismatch: "
            f"{key}={actual}, expected {expected}"
        )

print("Frozen model structure/parameters: PASS")


# ============================================================
# 12. COMPARE SCALERS
# ============================================================

new_scaler = reproduced_model.named_steps["scaler"]
old_scaler = frozen_model.named_steps["scaler"]

scaler_mean_max_abs_diff = float(
    np.max(np.abs(new_scaler.mean_ - old_scaler.mean_))
)

scaler_scale_max_abs_diff = float(
    np.max(np.abs(new_scaler.scale_ - old_scaler.scale_))
)

print("\nScaler comparison:")
print(
    f"max |mean difference|  = "
    f"{scaler_mean_max_abs_diff:.16g}"
)
print(
    f"max |scale difference| = "
    f"{scaler_scale_max_abs_diff:.16g}"
)


# ============================================================
# 13. DECISION-SCORE REPRODUCIBILITY
# ============================================================

new_val_scores = reproduced_model.decision_function(X_val)
old_val_scores = frozen_model.decision_function(X_val)

new_test_scores = reproduced_model.decision_function(X_test)
old_test_scores = frozen_model.decision_function(X_test)

val_score_max_abs_diff = float(
    np.max(np.abs(new_val_scores - old_val_scores))
)

test_score_max_abs_diff = float(
    np.max(np.abs(new_test_scores - old_test_scores))
)

val_prediction_match = float(
    np.mean(
        (new_val_scores >= FROZEN_THRESHOLD)
        ==
        (old_val_scores >= FROZEN_THRESHOLD)
    )
)

test_prediction_match = float(
    np.mean(
        (new_test_scores >= FROZEN_THRESHOLD)
        ==
        (old_test_scores >= FROZEN_THRESHOLD)
    )
)

print("\nDecision-score comparison against frozen V1:")
print(
    f"Validation max absolute difference: "
    f"{val_score_max_abs_diff:.16g}"
)
print(
    f"Test max absolute difference:       "
    f"{test_score_max_abs_diff:.16g}"
)
print(
    f"Validation prediction agreement:    "
    f"{val_prediction_match:.12f}"
)
print(
    f"Test prediction agreement:          "
    f"{test_prediction_match:.12f}"
)


# ============================================================
# 14. REPRODUCED METRICS
# ============================================================

validation_metrics = metrics_from_scores(
    y_val,
    new_val_scores,
    FROZEN_THRESHOLD,
)

test_metrics = metrics_from_scores(
    y_test,
    new_test_scores,
    FROZEN_THRESHOLD,
)

validation_pass, validation_comparison = compare_metrics(
    validation_metrics,
    EXPECTED_VALIDATION,
    rounded_expected=False,
)

test_pass, test_comparison = compare_metrics(
    test_metrics,
    EXPECTED_TEST,
    rounded_expected=True,
)


# ============================================================
# 15. STRUCTURAL MODEL COMPARISON
# ============================================================

new_clf = reproduced_model.named_steps["clf"]
old_clf = frozen_model.named_steps["clf"]

support_count_match = (
    len(new_clf.support_) == len(old_clf.support_)
)

n_support_match = np.array_equal(
    new_clf.n_support_,
    old_clf.n_support_,
)

classes_match = np.array_equal(
    new_clf.classes_,
    old_clf.classes_,
)


# ============================================================
# 16. OVERALL PASS/FAIL
# ============================================================

STRICT_SCORE_TOLERANCE = 1e-10

overall_pass = all(
    [
        validation_pass,
        test_pass,
        scaler_mean_max_abs_diff <= 1e-12,
        scaler_scale_max_abs_diff <= 1e-12,
        val_score_max_abs_diff <= STRICT_SCORE_TOLERANCE,
        test_score_max_abs_diff <= STRICT_SCORE_TOLERANCE,
        val_prediction_match == 1.0,
        test_prediction_match == 1.0,
        support_count_match,
        n_support_match,
        classes_match,
    ]
)


# ============================================================
# 17. SAVE REPRODUCED MODEL
# ============================================================

joblib.dump(
    reproduced_model,
    REPRO_MODEL_FILE,
)

reproduced_model_hash = sha256_file(REPRO_MODEL_FILE)
frozen_model_hash = sha256_file(FROZEN_MODEL_FILE)


# ============================================================
# 18. REPORT
# ============================================================

report = {
    "status": (
        "PASS_EXACT_REPRODUCTION"
        if overall_pass
        else "FAIL_REPRODUCTION_MISMATCH"
    ),
    "environment": {
        "python": sys.version,
        "scikit_learn": sklearn.__version__,
    },
    "frozen_configuration": {
        "feature_count": len(features),
        "features": features,
        "threshold": FROZEN_THRESHOLD,
        "svm": EXPECTED_PARAMS,
    },
    "split_rows": {
        "train": len(train_df),
        "validation": len(val_df),
        "test": len(test_df),
    },
    "group_overlap": overlaps,
    "scaler_comparison": {
        "max_abs_mean_difference": scaler_mean_max_abs_diff,
        "max_abs_scale_difference": scaler_scale_max_abs_diff,
    },
    "decision_score_comparison": {
        "validation_max_abs_difference": val_score_max_abs_diff,
        "test_max_abs_difference": test_score_max_abs_diff,
        "validation_prediction_agreement": val_prediction_match,
        "test_prediction_agreement": test_prediction_match,
    },
    "svm_structure_comparison": {
        "new_support_vector_count": int(len(new_clf.support_)),
        "frozen_support_vector_count": int(len(old_clf.support_)),
        "support_vector_count_match": bool(support_count_match),
        "new_n_support": [int(v) for v in new_clf.n_support_],
        "frozen_n_support": [int(v) for v in old_clf.n_support_],
        "n_support_match": bool(n_support_match),
        "classes_match": bool(classes_match),
    },
    "validation_metrics": validation_metrics,
    "validation_expected_comparison": validation_comparison,
    "test_metrics": test_metrics,
    "test_expected_comparison": test_comparison,
    "artifacts": {
        "frozen_model_sha256": frozen_model_hash,
        "reproduced_model_sha256": reproduced_model_hash,
        "reproduced_model_file": str(REPRO_MODEL_FILE),
    },
}

with open(REPORT_JSON, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)

lines = [
    "=" * 78,
    "V1 REPRODUCIBILITY REPORT",
    "=" * 78,
    "",
    f"STATUS: {report['status']}",
    "",
    f"scikit-learn: {sklearn.__version__}",
    f"Threshold: {FROZEN_THRESHOLD:.15f}",
    "",
    "SCALER",
    f"Max abs mean difference:  {scaler_mean_max_abs_diff:.16g}",
    f"Max abs scale difference: {scaler_scale_max_abs_diff:.16g}",
    "",
    "DECISION SCORES",
    f"Validation max abs difference: {val_score_max_abs_diff:.16g}",
    f"Test max abs difference:       {test_score_max_abs_diff:.16g}",
    f"Validation agreement:          {val_prediction_match:.12f}",
    f"Test agreement:                {test_prediction_match:.12f}",
    "",
    "VALIDATION",
    (
        f"Balanced Accuracy: "
        f"{validation_metrics['balanced_accuracy']:.12f}"
    ),
    f"ROC-AUC:           {validation_metrics['roc_auc']:.12f}",
    (
        f"TN/FP/FN/TP:       "
        f"{validation_metrics['TN']}/"
        f"{validation_metrics['FP']}/"
        f"{validation_metrics['FN']}/"
        f"{validation_metrics['TP']}"
    ),
    "",
    "TEST",
    (
        f"Balanced Accuracy: "
        f"{test_metrics['balanced_accuracy']:.12f}"
    ),
    f"ROC-AUC:           {test_metrics['roc_auc']:.12f}",
    (
        f"TN/FP/FN/TP:       "
        f"{test_metrics['TN']}/"
        f"{test_metrics['FP']}/"
        f"{test_metrics['FN']}/"
        f"{test_metrics['TP']}"
    ),
    "",
    "FROZEN MODEL SHA256",
    frozen_model_hash,
    "",
    "REPRODUCED MODEL SHA256",
    reproduced_model_hash,
    "",
    "NOTE:",
    (
        "A different .joblib SHA256 alone is not a reproduction failure; "
        "serialization metadata can differ. Numerical model behavior, "
        "scaler parameters, support-vector structure, and metrics are the "
        "primary checks."
    ),
    "",
    "=" * 78,
]

with open(REPORT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))


# ============================================================
# 19. DISPLAY RESULT
# ============================================================

print()
print("=" * 78)
print("REPRODUCTION RESULT")
print("=" * 78)
print()
print(report["status"])

print("\nValidation metrics:")
for key, value in validation_metrics.items():
    print(f"{key}: {value}")

print("\nTest metrics:")
for key, value in test_metrics.items():
    print(f"{key}: {value}")

print("\nReport:")
print(REPORT_JSON)

print("\nText report:")
print(REPORT_TXT)

print("\nReproduced model (separate; frozen V1 untouched):")
print(REPRO_MODEL_FILE)

if not overall_pass:
    raise RuntimeError(
        "\nV1 reproduction did not exactly match the frozen system. "
        "Open the JSON report and inspect the failed checks."
    )

print()
print("V1 exact reproduction: PASS")
