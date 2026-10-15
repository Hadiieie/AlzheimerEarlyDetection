from pathlib import Path
import json

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "V3_RESNET18_BASELINE"
    / "v3_resnet18_oof_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "V3_OOF_THRESHOLD"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SWEEP_FILE = OUTPUT_DIR / "v3_oof_threshold_sweep.csv"
SELECTED_FILE = OUTPUT_DIR / "v3_selected_threshold.json"
REPORT_JSON = OUTPUT_DIR / "v3_oof_threshold_report.json"
REPORT_TXT = OUTPUT_DIR / "v3_oof_threshold_report.txt"

TARGET = 0.90

TARGET_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "verymild_sensitivity",
    "mild_sensitivity",
    "nondemented_specificity",
]


def exhaustive_thresholds(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    unique_scores = np.unique(probabilities)
    unique_scores.sort()

    if len(unique_scores) == 1:
        return np.unique([
            np.nextafter(unique_scores[0], -np.inf),
            unique_scores[0],
            np.nextafter(unique_scores[0], np.inf),
            0.5,
        ])

    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0

    lower = np.nextafter(unique_scores[0], -np.inf)
    upper = np.nextafter(unique_scores[-1], np.inf)

    return np.unique(
        np.concatenate([
            np.array([lower]),
            midpoints,
            np.array([upper, 0.5]),
        ])
    )


def class_masks(df):
    values = (
        df["original_class"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

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
    probs,
    threshold,
    verymild_mask,
    mild_mask,
    normal_mask,
):
    pred = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    result = {
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
        "verymild_sensitivity": float(
            np.mean(pred[verymild_mask] == 1)
        ),
        "mild_sensitivity": float(
            np.mean(pred[mild_mask] == 1)
        ),
        "nondemented_specificity": float(
            np.mean(pred[normal_mask] == 0)
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    result["min_target_metric"] = float(
        min(result[name] for name in TARGET_METRICS)
    )

    result["metrics_above_90_count"] = int(
        sum(result[name] > TARGET for name in TARGET_METRICS)
    )

    result["all_target_metrics_above_90"] = bool(
        all(result[name] > TARGET for name in TARGET_METRICS)
    )

    return result


print()
print("=" * 96)
print("V3 OOF THRESHOLD OPTIMIZATION")
print("=" * 96)
print()
print("DATA POLICY")
print("-----------")
print("OOF predictions: LOADED")
print("old validation.csv: NOT LOADED")
print("old test.csv:       NOT LOADED")
print("CNN retraining:     NO")
print()

if not INPUT_FILE.exists():
    raise FileNotFoundError(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

required = {
    "target",
    "original_class",
    "resnet18_probability",
}

missing = sorted(required - set(df.columns))

if missing:
    raise ValueError(
        f"OOF file missing required columns: {missing}"
    )

if df["resnet18_probability"].isna().any():
    raise ValueError("NaN probabilities found.")

probs = df["resnet18_probability"].astype(float).to_numpy()
y_true = df["target"].astype(int).to_numpy()

if not set(np.unique(y_true)).issubset({0, 1}):
    raise ValueError("Target must contain only 0/1.")

if np.any((probs < 0.0) | (probs > 1.0)):
    raise ValueError("Probabilities must be within [0,1].")

verymild_mask, mild_mask, normal_mask = class_masks(df)

roc_auc = float(roc_auc_score(y_true, probs))
pr_auc = float(average_precision_score(y_true, probs))

print(f"OOF rows:          {len(df)}")
print(f"Normal:            {int((y_true == 0).sum())}")
print(f"Early_Demented:    {int((y_true == 1).sum())}")
print(f"VeryMild rows:     {int(verymild_mask.sum())}")
print(f"Mild rows:         {int(mild_mask.sum())}")
print(f"NonDemented rows:  {int(normal_mask.sum())}")
print()
print(f"Threshold-independent ROC-AUC: {roc_auc:.6f}")
print(f"Threshold-independent PR-AUC:  {pr_auc:.6f}")

thresholds = exhaustive_thresholds(probs)

print()
print(f"Non-redundant thresholds: {len(thresholds)}")

rows = []

for threshold in thresholds:
    result = evaluate_threshold(
        y_true=y_true,
        probs=probs,
        threshold=float(threshold),
        verymild_mask=verymild_mask,
        mild_mask=mild_mask,
        normal_mask=normal_mask,
    )
    result["roc_auc"] = roc_auc
    result["pr_auc"] = pr_auc
    rows.append(result)

sweep = pd.DataFrame(rows)
sweep.to_csv(SWEEP_FILE, index=False)

feasible = sweep[
    sweep["all_target_metrics_above_90"]
].copy()

if len(feasible) > 0:
    goal_status = "GOAL_ACHIEVED_WITH_THRESHOLD"

    feasible = feasible.sort_values(
        by=[
            "min_target_metric",
            "balanced_accuracy",
            "verymild_sensitivity",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    top_min = float(
        feasible.iloc[0]["min_target_metric"]
    )

    tied = feasible[
        np.isclose(
            feasible["min_target_metric"],
            top_min,
            atol=1e-12,
            rtol=0.0,
        )
    ].copy()

    top_ba = float(tied["balanced_accuracy"].max())
    tied = tied[
        np.isclose(
            tied["balanced_accuracy"],
            top_ba,
            atol=1e-12,
            rtol=0.0,
        )
    ].copy()

    top_vm = float(tied["verymild_sensitivity"].max())
    tied = tied[
        np.isclose(
            tied["verymild_sensitivity"],
            top_vm,
            atol=1e-12,
            rtol=0.0,
        )
    ].copy()

    tied["distance_from_0_5"] = (
        tied["threshold"] - 0.5
    ).abs()

    selected = (
        tied
        .sort_values(
            by=[
                "distance_from_0_5",
                "threshold",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .iloc[0]
    )

else:
    goal_status = "GOAL_NOT_ACHIEVED_BY_THRESHOLD_ALONE"

    candidates = sweep.copy()

    max_min = float(
        candidates["min_target_metric"].max()
    )

    candidates = candidates[
        np.isclose(
            candidates["min_target_metric"],
            max_min,
            atol=1e-12,
            rtol=0.0,
        )
    ].copy()

    max_count = int(
        candidates["metrics_above_90_count"].max()
    )

    candidates = candidates[
        candidates["metrics_above_90_count"] == max_count
    ].copy()

    max_ba = float(
        candidates["balanced_accuracy"].max()
    )

    candidates = candidates[
        np.isclose(
            candidates["balanced_accuracy"],
            max_ba,
            atol=1e-12,
            rtol=0.0,
        )
    ].copy()

    max_vm = float(
        candidates["verymild_sensitivity"].max()
    )

    candidates = candidates[
        np.isclose(
            candidates["verymild_sensitivity"],
            max_vm,
            atol=1e-12,
            rtol=0.0,
        )
    ].copy()

    candidates["distance_from_0_5"] = (
        candidates["threshold"] - 0.5
    ).abs()

    selected = (
        candidates
        .sort_values(
            by=[
                "distance_from_0_5",
                "threshold",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .iloc[0]
    )

selected_threshold = float(selected["threshold"])

below_90 = {
    metric: float(selected[metric])
    for metric in TARGET_METRICS
    if float(selected[metric]) <= TARGET
}

selected_metrics = {}
for key, value in selected.items():
    if key == "distance_from_0_5":
        continue

    if isinstance(value, (bool, np.bool_)):
        selected_metrics[key] = bool(value)
    elif isinstance(value, (int, np.integer)):
        selected_metrics[key] = int(value)
    elif pd.isna(value):
        selected_metrics[key] = None
    else:
        selected_metrics[key] = float(value)

baseline_row = sweep.iloc[
    (sweep["threshold"] - 0.5).abs().argmin()
]

baseline_metrics = {
    metric: float(baseline_row[metric])
    for metric in TARGET_METRICS
}

print()
print("=" * 96)
print("THRESHOLD SEARCH RESULT")
print("=" * 96)
print()
print(f"Status: {goal_status}")
print(f"Selected threshold: {selected_threshold:.12f}")
print()

for metric in TARGET_METRICS:
    value = float(selected[metric])
    marker = "PASS" if value > TARGET else "BELOW 90%"
    print(
        f"{metric:28s} "
        f"{value * 100:8.3f}%   "
        f"{marker}"
    )

print()
print(f"ROC-AUC: {roc_auc * 100:.3f}%")
print(f"PR-AUC:  {pr_auc * 100:.3f}%")
print(
    f"Weakest target metric: "
    f"{float(selected['min_target_metric']) * 100:.3f}%"
)
print(
    f"Metrics >90%: "
    f"{int(selected['metrics_above_90_count'])}"
    f"/{len(TARGET_METRICS)}"
)

if below_90:
    print()
    print("Still at/below 90%:")
    for metric, value in below_90.items():
        print(
            f"  {metric}: "
            f"{value * 100:.3f}%"
        )
else:
    print()
    print(
        "ALL requested threshold-dependent "
        "metrics are strictly above 90%."
    )

payload = {
    "status": goal_status,
    "development_only": True,
    "data_policy": {
        "oof_predictions_loaded": True,
        "old_validation_loaded": False,
        "old_test_loaded": False,
        "cnn_retrained": False,
    },
    "target_requirement": (
        "Strictly greater than 0.90 for all listed "
        "threshold-dependent target metrics."
    ),
    "target_metrics": TARGET_METRICS,
    "selected_threshold": selected_threshold,
    "selected_metrics": selected_metrics,
    "threshold_independent_metrics": {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    },
    "baseline_threshold_0_5_metrics": baseline_metrics,
    "metrics_at_or_below_90_at_selected_threshold": below_90,
    "feasible_threshold_count": int(len(feasible)),
    "thresholds_checked": int(len(thresholds)),
    "important": (
        "This is OOF development-only threshold selection. "
        "It is not final patient-level or external-test performance."
    ),
}

with open(SELECTED_FILE, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=4)

with open(REPORT_JSON, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=4)

lines = [
    "=" * 96,
    "V3 OOF THRESHOLD OPTIMIZATION",
    "=" * 96,
    "",
    f"STATUS: {goal_status}",
    "",
    f"Selected threshold: {selected_threshold:.12f}",
    "",
]

for metric in TARGET_METRICS:
    value = float(selected[metric])
    lines.append(
        f"{metric}: {value * 100:.3f}%"
    )

lines.extend([
    "",
    f"ROC-AUC: {roc_auc * 100:.3f}%",
    f"PR-AUC: {pr_auc * 100:.3f}%",
    (
        "Weakest target metric: "
        f"{float(selected['min_target_metric']) * 100:.3f}%"
    ),
    (
        "Metrics strictly >90%: "
        f"{int(selected['metrics_above_90_count'])}"
        f"/{len(TARGET_METRICS)}"
    ),
    "",
])

if below_90:
    lines.append("Still at/below 90%:")
    for metric, value in below_90.items():
        lines.append(
            f"  {metric}: {value * 100:.3f}%"
        )
else:
    lines.append(
        "ALL requested threshold-dependent metrics "
        "are strictly above 90%."
    )

lines.extend([
    "",
    "DEVELOPMENT ONLY",
    "Old validation/test were NOT loaded.",
    "",
])

REPORT_TXT.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print()
print("Saved:")
print(SWEEP_FILE)
print(SELECTED_FILE)
print(REPORT_JSON)
print(REPORT_TXT)
