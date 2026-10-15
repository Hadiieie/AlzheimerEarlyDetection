from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ============================================================
# V2 VALIDATION THRESHOLD SELECTION
#
# Locked from TRAINING-ONLY group-aware CV:
#   C = 50
#   gamma = 0.06
#   kernel = rbf
#   class_weight = balanced
#
# DATA POLICY:
#   train.csv      -> fit model
#   validation.csv -> select threshold
#   test.csv       -> NOT LOADED
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLIT_DIR = PROJECT_ROOT / "results" / "GROUP_AWARE_SPLIT"
FINAL_MODEL_DIR = PROJECT_ROOT / "results" / "FINAL_MODEL"
CV_DIR = PROJECT_ROOT / "results" / "V2_SVM_GROUP_CV"
OUTPUT_DIR = PROJECT_ROOT / "results" / "V2_VALIDATION_THRESHOLD"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = SPLIT_DIR / "train.csv"
VAL_FILE = SPLIT_DIR / "validation.csv"
FEATURE_FILE = FINAL_MODEL_DIR / "final_features.txt"
CV_BEST_PARAMS_FILE = CV_DIR / "best_hyperparameters.json"

MODEL_FILE = OUTPUT_DIR / "v2_candidate_model_pre_freeze.joblib"
SWEEP_FILE = OUTPUT_DIR / "v2_validation_threshold_sweep.csv"
THRESHOLD_FILE = OUTPUT_DIR / "v2_selected_threshold.json"
REPORT_FILE = OUTPUT_DIR / "v2_validation_threshold_report.json"
TEXT_REPORT_FILE = OUTPUT_DIR / "v2_validation_threshold_report.txt"

LOCKED_C = 50.0
LOCKED_GAMMA = 0.06


def load_features(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def detect_original_class_column(df):
    candidates = [
        "original_class",
        "original_label",
        "source_class",
        "class_name",
        "class",
        "label_name",
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def validate_split(name, df, features):
    if "target" not in df.columns:
        raise ValueError(f"{name}: missing target column")

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing features: {missing}")

    X = df[features].to_numpy(dtype=float)

    if np.isnan(X).any():
        raise ValueError(f"{name}: NaN found in feature matrix")

    if np.isinf(X).any():
        raise ValueError(f"{name}: Inf found in feature matrix")

    targets = set(pd.unique(df["target"]))
    if not targets.issubset({0, 1}):
        raise ValueError(f"{name}: target must contain only 0/1")


def subgroup_masks(df, class_col):
    if class_col is None:
        return None, None, None

    values = df[class_col].astype(str).str.strip().str.lower()

    verymild = values.isin({
        "verymilddemented",
        "very mild demented",
        "very_mild_demented",
    }).to_numpy()

    mild = values.isin({
        "milddemented",
        "mild demented",
        "mild_demented",
    }).to_numpy()

    normal = values.isin({
        "nondemented",
        "non demented",
        "non_demented",
        "normal",
    }).to_numpy()

    return verymild, mild, normal


def evaluate_threshold(
    y_true,
    scores,
    threshold,
    verymild_mask,
    mild_mask,
    normal_mask,
):
    pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    verymild_sensitivity = None
    mild_sensitivity = None
    nondemented_specificity = None

    if verymild_mask is not None and verymild_mask.any():
        verymild_sensitivity = float(
            np.mean(pred[verymild_mask] == 1)
        )

    if mild_mask is not None and mild_mask.any():
        mild_sensitivity = float(
            np.mean(pred[mild_mask] == 1)
        )

    if normal_mask is not None and normal_mask.any():
        nondemented_specificity = float(
            np.mean(pred[normal_mask] == 0)
        )

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, pred)
        ),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(
            precision_score(
                y_true,
                pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "verymild_sensitivity": verymild_sensitivity,
        "mild_sensitivity": mild_sensitivity,
        "nondemented_specificity": nondemented_specificity,
        "sens_spec_gap": float(abs(sensitivity - specificity)),
        "youden_j": float(sensitivity + specificity - 1.0),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def exhaustive_thresholds(scores):
    scores = np.asarray(scores, dtype=float)
    unique_scores = np.unique(scores)
    unique_scores.sort()

    if len(unique_scores) == 1:
        return np.unique([
            np.nextafter(unique_scores[0], -np.inf),
            unique_scores[0],
            np.nextafter(unique_scores[0], np.inf),
            0.0,
        ])

    midpoints = (
        unique_scores[:-1] + unique_scores[1:]
    ) / 2.0

    lower = np.nextafter(unique_scores[0], -np.inf)
    upper = np.nextafter(unique_scores[-1], np.inf)

    return np.unique(
        np.concatenate([
            np.array([lower]),
            midpoints,
            np.array([upper, 0.0]),
        ])
    )


print()
print("=" * 90)
print("V2 VALIDATION THRESHOLD SELECTION")
print("=" * 90)
print("Locked hyperparameters:")
print(f"C = {LOCKED_C}")
print(f"gamma = {LOCKED_GAMMA}")
print()
print("DATA POLICY")
print("train.csv      : loaded")
print("validation.csv : loaded")
print("test.csv       : NOT LOADED")
print()

for path in [TRAIN_FILE, VAL_FILE, FEATURE_FILE]:
    if not path.exists():
        raise FileNotFoundError(path)

# Verify CV-selected parameters if file exists.
if CV_BEST_PARAMS_FILE.exists():
    with open(CV_BEST_PARAMS_FILE, "r", encoding="utf-8") as f:
        cv_best = json.load(f)

    cv_c = float(cv_best["C"])
    cv_gamma = float(cv_best["gamma"])

    if not np.isclose(cv_c, LOCKED_C, atol=1e-15, rtol=0):
        raise RuntimeError(
            f"CV-selected C={cv_c}, expected locked C={LOCKED_C}"
        )

    if not np.isclose(
        cv_gamma,
        LOCKED_GAMMA,
        atol=1e-15,
        rtol=0,
    ):
        raise RuntimeError(
            "CV-selected gamma does not match locked gamma"
        )

    print("Training-CV parameter verification: PASS")

features = load_features(FEATURE_FILE)

if len(features) != 30:
    raise ValueError(
        f"Expected 30 frozen features, found {len(features)}"
    )

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)

validate_split("train", train_df, features)
validate_split("validation", val_df, features)

class_col = detect_original_class_column(val_df)

print(f"Train rows:      {len(train_df)}")
print(f"Validation rows: {len(val_df)}")
print(f"Feature count:   {len(features)}")
print(f"Original class column: {class_col}")

X_train = train_df[features]
y_train = train_df["target"].astype(int).to_numpy()

X_val = val_df[features]
y_val = val_df["target"].astype(int).to_numpy()

verymild_mask, mild_mask, normal_mask = subgroup_masks(
    val_df,
    class_col,
)

print()
print("=" * 90)
print("FITTING LOCKED V2 MODEL")
print("=" * 90)

model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "clf",
        SVC(
            kernel="rbf",
            C=LOCKED_C,
            gamma=LOCKED_GAMMA,
            class_weight="balanced",
            probability=False,
            cache_size=4000,
        ),
    ),
])

start = time.perf_counter()
model.fit(X_train, y_train)
fit_seconds = time.perf_counter() - start

print(f"Training complete in {fit_seconds:.1f} seconds")

joblib.dump(model, MODEL_FILE)

scores = model.decision_function(X_val)

roc_auc = float(roc_auc_score(y_val, scores))
pr_auc = float(average_precision_score(y_val, scores))

default_metrics = evaluate_threshold(
    y_val,
    scores,
    0.0,
    verymild_mask,
    mild_mask,
    normal_mask,
)

print()
print("=" * 90)
print("DEFAULT THRESHOLD = 0.0")
print("=" * 90)
print(
    f"Balanced Accuracy: {default_metrics['balanced_accuracy']:.6f}"
)
print(f"Accuracy:          {default_metrics['accuracy']:.6f}")
print(f"Sensitivity:       {default_metrics['sensitivity']:.6f}")
print(f"Specificity:       {default_metrics['specificity']:.6f}")
print(f"ROC-AUC:           {roc_auc:.6f}")
print(f"PR-AUC:            {pr_auc:.6f}")

if default_metrics["verymild_sensitivity"] is not None:
    print(
        "VeryMild Sens.:     "
        f"{default_metrics['verymild_sensitivity']:.6f}"
    )

if default_metrics["mild_sensitivity"] is not None:
    print(
        "Mild Sens.:         "
        f"{default_metrics['mild_sensitivity']:.6f}"
    )

if default_metrics["nondemented_specificity"] is not None:
    print(
        "NonDemented Spec.:  "
        f"{default_metrics['nondemented_specificity']:.6f}"
    )

print()
print("=" * 90)
print("EXHAUSTIVE VALIDATION THRESHOLD SEARCH")
print("=" * 90)

thresholds = exhaustive_thresholds(scores)

rows = []

for threshold in thresholds:
    row = evaluate_threshold(
        y_val,
        scores,
        float(threshold),
        verymild_mask,
        mild_mask,
        normal_mask,
    )
    row["roc_auc"] = roc_auc
    row["pr_auc"] = pr_auc
    rows.append(row)

sweep_df = pd.DataFrame(rows)
sweep_df.to_csv(SWEEP_FILE, index=False)

# Selection rule:
# 1) max balanced accuracy
# 2) higher VeryMild sensitivity
# 3) smaller sensitivity-specificity gap
# 4) threshold closer to zero

best_ba = float(sweep_df["balanced_accuracy"].max())

candidates = sweep_df[
    np.isclose(
        sweep_df["balanced_accuracy"],
        best_ba,
        atol=1e-12,
        rtol=0.0,
    )
].copy()

if candidates["verymild_sensitivity"].notna().any():
    best_vm = float(
        candidates["verymild_sensitivity"].max()
    )
    candidates = candidates[
        np.isclose(
            candidates["verymild_sensitivity"],
            best_vm,
            atol=1e-12,
            rtol=0.0,
        )
    ].copy()

best_gap = float(candidates["sens_spec_gap"].min())

candidates = candidates[
    np.isclose(
        candidates["sens_spec_gap"],
        best_gap,
        atol=1e-12,
        rtol=0.0,
    )
].copy()

candidates["abs_threshold"] = candidates["threshold"].abs()

selected = (
    candidates
    .sort_values(
        by=["abs_threshold", "threshold"],
        ascending=[True, True],
    )
    .iloc[0]
)

selected_threshold = float(selected["threshold"])

selected_metrics = {
    k: (
        None
        if pd.isna(v)
        else (
            int(v)
            if k in {"TN", "FP", "FN", "TP"}
            else float(v)
        )
    )
    for k, v in selected.items()
    if k != "abs_threshold"
}

print()
print("=" * 90)
print("SELECTED V2 VALIDATION THRESHOLD")
print("=" * 90)
print(f"Threshold:         {selected_threshold:.15f}")
print(
    "Balanced Accuracy:"
    f" {selected_metrics['balanced_accuracy']:.6f}"
)
print(f"Accuracy:          {selected_metrics['accuracy']:.6f}")
print(
    f"Sensitivity:       {selected_metrics['sensitivity']:.6f}"
)
print(
    f"Specificity:       {selected_metrics['specificity']:.6f}"
)
print(f"Precision:         {selected_metrics['precision']:.6f}")
print(f"F1:                {selected_metrics['f1']:.6f}")
print(f"ROC-AUC:           {roc_auc:.6f}")
print(f"PR-AUC:            {pr_auc:.6f}")

if selected_metrics["verymild_sensitivity"] is not None:
    print(
        "VeryMild Sens.:     "
        f"{selected_metrics['verymild_sensitivity']:.6f}"
    )

if selected_metrics["mild_sensitivity"] is not None:
    print(
        "Mild Sens.:         "
        f"{selected_metrics['mild_sensitivity']:.6f}"
    )

if selected_metrics["nondemented_specificity"] is not None:
    print(
        "NonDemented Spec.:  "
        f"{selected_metrics['nondemented_specificity']:.6f}"
    )

print(
    "TN / FP / FN / TP:"
    f" {selected_metrics['TN']} /"
    f" {selected_metrics['FP']} /"
    f" {selected_metrics['FN']} /"
    f" {selected_metrics['TP']}"
)

threshold_payload = {
    "status": "V2_VALIDATION_SELECTED_NOT_TESTED",
    "selected_threshold": selected_threshold,
    "locked_model": {
        "kernel": "rbf",
        "C": LOCKED_C,
        "gamma": LOCKED_GAMMA,
        "class_weight": "balanced",
        "feature_count": len(features),
    },
    "selection_rule": {
        "primary": "maximum validation balanced accuracy",
        "tie_break_1": "higher VeryMildDemented sensitivity",
        "tie_break_2": (
            "smaller absolute sensitivity-specificity gap"
        ),
        "tie_break_3": "threshold closest to zero",
    },
    "validation_default_threshold_metrics": default_metrics,
    "validation_selected_threshold_metrics": selected_metrics,
    "validation_roc_auc": roc_auc,
    "validation_pr_auc": pr_auc,
    "threshold_candidates": int(len(thresholds)),
    "test_set_loaded": False,
}

with open(THRESHOLD_FILE, "w", encoding="utf-8") as f:
    json.dump(threshold_payload, f, indent=4)

report = {
    "data_policy": {
        "train_loaded": True,
        "validation_loaded": True,
        "test_loaded": False,
    },
    "train_rows": int(len(train_df)),
    "validation_rows": int(len(val_df)),
    "feature_count": int(len(features)),
    "fit_seconds": float(fit_seconds),
    "selected_threshold": threshold_payload,
}

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)

text = f'''
V2 VALIDATION THRESHOLD SELECTION
=================================

LOCKED MODEL
C = {LOCKED_C}
gamma = {LOCKED_GAMMA}
kernel = rbf
class_weight = balanced

DATA POLICY
Train:      used for model fitting
Validation: used for threshold selection
Test:       NOT LOADED

DEFAULT THRESHOLD = 0.0
Balanced Accuracy = {default_metrics["balanced_accuracy"]:.12f}
VeryMild Sensitivity = {default_metrics["verymild_sensitivity"]}

SELECTED THRESHOLD
Threshold = {selected_threshold:.15f}
Balanced Accuracy = {selected_metrics["balanced_accuracy"]:.12f}
Accuracy = {selected_metrics["accuracy"]:.12f}
Sensitivity = {selected_metrics["sensitivity"]:.12f}
Specificity = {selected_metrics["specificity"]:.12f}
VeryMild Sensitivity = {selected_metrics["verymild_sensitivity"]}
Mild Sensitivity = {selected_metrics["mild_sensitivity"]}
NonDemented Specificity = {selected_metrics["nondemented_specificity"]}
ROC-AUC = {roc_auc:.12f}
PR-AUC = {pr_auc:.12f}
TN/FP/FN/TP = {selected_metrics["TN"]}/{selected_metrics["FP"]}/{selected_metrics["FN"]}/{selected_metrics["TP"]}

STATUS
V2 hyperparameters are locked.
V2 validation threshold is selected.
V2 has NOT been evaluated on the test set yet.
'''

TEXT_REPORT_FILE.write_text(text.strip() + "\n", encoding="utf-8")

print()
print("=" * 90)
print("V2 VALIDATION STAGE COMPLETE")
print("=" * 90)
print(f"Locked C     = {LOCKED_C}")
print(f"Locked gamma = {LOCKED_GAMMA}")
print(f"Threshold    = {selected_threshold:.15f}")
print()
print("TEST SET REMAINS UNTOUCHED.")
print("Do not change C/gamma after this stage.")
print()
print("Saved:")
print(MODEL_FILE)
print(SWEEP_FILE)
print(THRESHOLD_FILE)
print(REPORT_FILE)
print(TEXT_REPORT_FILE)
