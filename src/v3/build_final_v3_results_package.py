from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    matthews_corrcoef,
)


# ============================================================
# FINAL V3 RESULTS PACKAGE
# Alzheimer's MRI Early Detection
#
# IMPORTANT
# ---------
# - NO model training
# - NO threshold tuning
# - NO model modification
# - Reads already-generated frozen/development artifacts only
#
# V3 headline results are leakage-safe GROUP-AWARE OOF
# development performance, NOT external validation.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

V3_OOF_FILE = (
    RESULTS_DIR
    / "V3_RESNET18_BASELINE"
    / "v3_resnet18_oof_predictions.csv"
)

V3_THRESHOLD_FILE = (
    RESULTS_DIR
    / "V3_OOF_THRESHOLD"
    / "v3_selected_threshold.json"
)

V3_HISTORY_FILE = (
    RESULTS_DIR
    / "V3_RESNET18_BASELINE"
    / "training_history.csv"
)

V2_REPORT_FILE = (
    RESULTS_DIR
    / "FINAL_V2_TEST"
    / "final_v2_test_report.json"
)

OUTPUT_DIR = (
    RESULTS_DIR
    / "FINAL_V3_RESULTS_PACKAGE"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# OUTPUT FILES
# ============================================================

METRICS_CSV = OUTPUT_DIR / "final_v3_primary_metrics.csv"
METRICS_JSON = OUTPUT_DIR / "final_v3_primary_metrics.json"
COMPARISON_CSV = OUTPUT_DIR / "v1_v2_v3_comparison.csv"
SUMMARY_TXT = OUTPUT_DIR / "FINAL_RESULTS_SUMMARY.txt"

CM_PNG = OUTPUT_DIR / "v3_confusion_matrix.png"
ROC_PNG = OUTPUT_DIR / "v3_roc_curve.png"
PR_PNG = OUTPUT_DIR / "v3_precision_recall_curve.png"
COMPARE_PNG = OUTPUT_DIR / "v1_v2_v3_comparison.png"
LOSS_PNG = OUTPUT_DIR / "v3_training_loss.png"
BA_PNG = OUTPUT_DIR / "v3_validation_balanced_accuracy.png"


# ============================================================
# VALIDATE INPUT FILES
# ============================================================

required_files = [
    V3_OOF_FILE,
    V3_THRESHOLD_FILE,
]

for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file missing:\n{path}"
        )


print()
print("=" * 90)
print("BUILD FINAL V3 RESULTS PACKAGE")
print("=" * 90)
print()

print("Model training:       NO")
print("Threshold tuning:     NO")
print("Model modification:   NO")
print("Existing results only: YES")
print()


# ============================================================
# LOAD V3 OOF DATA
# ============================================================

oof = pd.read_csv(
    V3_OOF_FILE
)

required_columns = {
    "target",
    "original_class",
    "resnet18_probability",
}

missing = sorted(
    required_columns
    - set(oof.columns)
)

if missing:

    raise ValueError(
        f"V3 OOF file missing columns: {missing}"
    )


y_true = (
    oof["target"]
    .astype(int)
    .to_numpy()
)

probabilities = (
    oof["resnet18_probability"]
    .astype(float)
    .to_numpy()
)


if len(oof) == 0:

    raise RuntimeError(
        "V3 OOF predictions file is empty."
    )


if np.isnan(probabilities).any():

    raise RuntimeError(
        "NaN probabilities detected."
    )


if np.any(
    (probabilities < 0)
    |
    (probabilities > 1)
):

    raise RuntimeError(
        "Probabilities outside [0,1]."
    )


# ============================================================
# LOAD EXACT LOCKED V3 THRESHOLD
# ============================================================

with open(
    V3_THRESHOLD_FILE,
    "r",
    encoding="utf-8",
) as f:

    threshold_info = json.load(f)


threshold = float(
    threshold_info[
        "selected_threshold"
    ]
)


print(
    f"Exact V3 threshold: {threshold:.17f}"
)


# ============================================================
# FINAL PREDICTIONS AT LOCKED THRESHOLD
# ============================================================

predictions = (
    probabilities
    >= threshold
).astype(int)


tn, fp, fn, tp = confusion_matrix(
    y_true,
    predictions,
    labels=[0, 1],
).ravel()


accuracy = float(
    accuracy_score(
        y_true,
        predictions,
    )
)

balanced_accuracy = float(
    balanced_accuracy_score(
        y_true,
        predictions,
    )
)

sensitivity = float(
    recall_score(
        y_true,
        predictions,
        pos_label=1,
        zero_division=0,
    )
)

specificity = float(
    tn / (tn + fp)
)

precision = float(
    precision_score(
        y_true,
        predictions,
        pos_label=1,
        zero_division=0,
    )
)

f1 = float(
    f1_score(
        y_true,
        predictions,
        pos_label=1,
        zero_division=0,
    )
)

roc_auc = float(
    roc_auc_score(
        y_true,
        probabilities,
    )
)

pr_auc = float(
    average_precision_score(
        y_true,
        probabilities,
    )
)

mcc = float(
    matthews_corrcoef(
        y_true,
        predictions,
    )
)


# ============================================================
# SUBGROUP PERFORMANCE
# ============================================================

classes = (
    oof["original_class"]
    .astype(str)
    .str.strip()
    .str.lower()
)


very_mild_mask = classes.isin({
    "verymilddemented",
    "very mild demented",
    "very_mild_demented",
}).to_numpy()


mild_mask = classes.isin({
    "milddemented",
    "mild demented",
    "mild_demented",
}).to_numpy()


normal_mask = classes.isin({
    "nondemented",
    "non demented",
    "non_demented",
    "normal",
}).to_numpy()


very_mild_sensitivity = float(
    np.mean(
        predictions[
            very_mild_mask
        ] == 1
    )
)


mild_sensitivity = float(
    np.mean(
        predictions[
            mild_mask
        ] == 1
    )
)


normal_specificity = float(
    np.mean(
        predictions[
            normal_mask
        ] == 0
    )
)


# ============================================================
# METRICS TABLE
# ============================================================

metrics = {
    "accuracy": accuracy,
    "balanced_accuracy": balanced_accuracy,
    "sensitivity": sensitivity,
    "specificity": specificity,
    "precision": precision,
    "f1": f1,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "mcc": mcc,
    "verymild_sensitivity": very_mild_sensitivity,
    "mild_sensitivity": mild_sensitivity,
    "nondemented_specificity": normal_specificity,
    "TN": int(tn),
    "FP": int(fp),
    "FN": int(fn),
    "TP": int(tp),
    "threshold": threshold,
    "oof_rows": int(len(oof)),
}


metrics_df = pd.DataFrame(
    [
        {
            "metric": key,
            "value": value,
        }
        for key, value in metrics.items()
    ]
)


metrics_df.to_csv(
    METRICS_CSV,
    index=False,
)


with open(
    METRICS_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        {
            "model": "V3 ResNet18",
            "status": "PRIMARY_PROJECT_MODEL",
            "evaluation_scope": (
                "Leakage-safe group-aware "
                "out-of-fold development evaluation"
            ),
            "external_validation": False,
            "metrics": metrics,
        },
        f,
        indent=4,
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = np.array([
    [tn, fp],
    [fn, tp],
])


fig, ax = plt.subplots(
    figsize=(7, 6)
)

image = ax.imshow(
    cm,
    interpolation="nearest",
)


ax.set_title(
    "V3 ResNet18 Confusion Matrix\n"
    "Group-Aware OOF Evaluation",
    fontsize=14,
    pad=15,
)


ax.set_xticks(
    [0, 1],
    labels=[
        "Normal",
        "Early Demented",
    ],
)


ax.set_yticks(
    [0, 1],
    labels=[
        "Normal",
        "Early Demented",
    ],
)


ax.set_xlabel(
    "Predicted Class"
)

ax.set_ylabel(
    "True Class"
)


threshold_cm = (
    cm.max()
    / 2.0
)


for i in range(2):

    for j in range(2):

        value = int(
            cm[i, j]
        )

        ax.text(
            j,
            i,
            str(value),
            ha="center",
            va="center",
            fontsize=15,
            color=(
                "white"
                if value > threshold_cm
                else "black"
            ),
        )


fig.colorbar(
    image,
    ax=ax,
    fraction=0.046,
    pad=0.04,
)


fig.tight_layout()

fig.savefig(
    CM_PNG,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, _ = roc_curve(
    y_true,
    probabilities,
)


fig, ax = plt.subplots(
    figsize=(7, 6)
)


ax.plot(
    fpr,
    tpr,
    linewidth=2,
    label=(
        f"V3 ResNet18 "
        f"(AUC = {roc_auc:.3f})"
    ),
)


ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
    label="Random classifier",
)


ax.set_xlabel(
    "False Positive Rate"
)

ax.set_ylabel(
    "True Positive Rate"
)

ax.set_title(
    "V3 ResNet18 ROC Curve\n"
    "Group-Aware OOF Evaluation"
)

ax.set_xlim(
    0,
    1
)

ax.set_ylim(
    0,
    1.02
)

ax.grid(
    alpha=0.25
)

ax.legend(
    loc="lower right"
)


fig.tight_layout()

fig.savefig(
    ROC_PNG,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

pr_precision, pr_recall, _ = (
    precision_recall_curve(
        y_true,
        probabilities,
    )
)


baseline = float(
    np.mean(
        y_true
    )
)


fig, ax = plt.subplots(
    figsize=(7, 6)
)


ax.plot(
    pr_recall,
    pr_precision,
    linewidth=2,
    label=(
        f"V3 ResNet18 "
        f"(AP = {pr_auc:.3f})"
    ),
)


ax.axhline(
    baseline,
    linestyle="--",
    linewidth=1,
    label=(
        f"Class prevalence "
        f"({baseline:.3f})"
    ),
)


ax.set_xlabel(
    "Recall"
)

ax.set_ylabel(
    "Precision"
)

ax.set_title(
    "V3 ResNet18 Precision-Recall Curve\n"
    "Group-Aware OOF Evaluation"
)

ax.set_xlim(
    0,
    1
)

ax.set_ylim(
    0,
    1.02
)

ax.grid(
    alpha=0.25
)

ax.legend(
    loc="lower left"
)


fig.tight_layout()

fig.savefig(
    PR_PNG,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# LOAD V1 + V2 FOR PROJECT-PROGRESSION COMPARISON
# ============================================================

comparison_rows = []


if V2_REPORT_FILE.exists():

    with open(
        V2_REPORT_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        v2_report = json.load(f)


    v1_metrics = (
        v2_report
        .get("V1", {})
        .get("metrics", {})
    )


    v2_metrics = (
        v2_report
        .get("V2", {})
        .get("metrics", {})
    )


    if v1_metrics:

        comparison_rows.append({
            "model": "V1 SVM",
            "evaluation_scope": (
                "Existing internal held-out test"
            ),
            "accuracy":
                float(v1_metrics["accuracy"]),
            "balanced_accuracy":
                float(
                    v1_metrics[
                        "balanced_accuracy"
                    ]
                ),
            "sensitivity":
                float(
                    v1_metrics[
                        "sensitivity"
                    ]
                ),
            "specificity":
                float(
                    v1_metrics[
                        "specificity"
                    ]
                ),
            "precision":
                float(
                    v1_metrics[
                        "precision"
                    ]
                ),
            "f1":
                float(
                    v1_metrics[
                        "f1"
                    ]
                ),
            "roc_auc":
                float(
                    v1_metrics[
                        "roc_auc"
                    ]
                ),
            "pr_auc":
                float(
                    v1_metrics[
                        "pr_auc"
                    ]
                ),
        })


    if v2_metrics:

        comparison_rows.append({
            "model": "V2 SVM",
            "evaluation_scope": (
                "Existing internal held-out test"
            ),
            "accuracy":
                float(v2_metrics["accuracy"]),
            "balanced_accuracy":
                float(
                    v2_metrics[
                        "balanced_accuracy"
                    ]
                ),
            "sensitivity":
                float(
                    v2_metrics[
                        "sensitivity"
                    ]
                ),
            "specificity":
                float(
                    v2_metrics[
                        "specificity"
                    ]
                ),
            "precision":
                float(
                    v2_metrics[
                        "precision"
                    ]
                ),
            "f1":
                float(
                    v2_metrics[
                        "f1"
                    ]
                ),
            "roc_auc":
                float(
                    v2_metrics[
                        "roc_auc"
                    ]
                ),
            "pr_auc":
                float(
                    v2_metrics[
                        "pr_auc"
                    ]
                ),
        })


comparison_rows.append({

    "model":
        "V3 ResNet18",

    "evaluation_scope":
        (
            "Group-aware OOF development"
        ),

    "accuracy":
        accuracy,

    "balanced_accuracy":
        balanced_accuracy,

    "sensitivity":
        sensitivity,

    "specificity":
        specificity,

    "precision":
        precision,

    "f1":
        f1,

    "roc_auc":
        roc_auc,

    "pr_auc":
        pr_auc,
})


comparison = pd.DataFrame(
    comparison_rows
)


comparison.to_csv(
    COMPARISON_CSV,
    index=False,
)


# ============================================================
# V1 / V2 / V3 COMPARISON FIGURE
# ============================================================

plot_metrics = [
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "f1",
    "roc_auc",
]


labels = [
    "Accuracy",
    "Balanced\nAccuracy",
    "Sensitivity",
    "Specificity",
    "F1",
    "ROC-AUC",
]


x = np.arange(
    len(plot_metrics)
)


width = (
    0.8
    /
    len(comparison)
)


fig, ax = plt.subplots(
    figsize=(12, 6)
)


for index, row in comparison.iterrows():

    values = [
        float(
            row[m]
        )
        * 100
        for m in plot_metrics
    ]

    offset = (
        index
        -
        (len(comparison) - 1) / 2
    ) * width

    ax.bar(
        x + offset,
        values,
        width,
        label=row["model"],
    )


ax.axhline(
    90,
    linestyle="--",
    linewidth=1,
    label="90% reference",
)


ax.set_xticks(
    x,
    labels
)

ax.set_ylabel(
    "Performance (%)"
)

ax.set_ylim(
    0,
    105
)

ax.set_title(
    "Project Model Performance Progression"
)

ax.legend(
    loc="lower right"
)

ax.grid(
    axis="y",
    alpha=0.2
)


fig.text(
    0.5,
    0.01,
    (
        "Important: V1/V2 use the existing internal "
        "held-out test; V3 uses leakage-safe "
        "group-aware OOF development evaluation. "
        "They are not identical evaluation protocols."
    ),
    ha="center",
    fontsize=8,
)


fig.tight_layout(
    rect=[
        0,
        0.05,
        1,
        1,
    ]
)


fig.savefig(
    COMPARE_PNG,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# TRAINING CURVES
# ============================================================

if V3_HISTORY_FILE.exists():

    history = pd.read_csv(
        V3_HISTORY_FILE
    )


    required_history = {
        "fold",
        "stage",
        "epoch",
        "train_loss",
        "val_balanced_accuracy",
    }


    if required_history.issubset(
        history.columns
    ):

        history = history.copy()

        history[
            "training_step"
        ] = (
            history
            .groupby("fold")
            .cumcount()
            + 1
        )


        # --------------------------------------------
        # TRAINING LOSS
        # --------------------------------------------

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )


        for fold, group in history.groupby(
            "fold"
        ):

            group = group.sort_values(
                "training_step"
            )

            ax.plot(
                group[
                    "training_step"
                ],
                group[
                    "train_loss"
                ],
                marker="o",
                label=f"Fold {fold}",
            )


        ax.set_xlabel(
            "Training Step"
        )

        ax.set_ylabel(
            "Training Loss"
        )

        ax.set_title(
            "V3 ResNet18 Training Loss"
        )

        ax.grid(
            alpha=0.25
        )

        ax.legend()


        fig.tight_layout()

        fig.savefig(
            LOSS_PNG,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


        # --------------------------------------------
        # VALIDATION BALANCED ACCURACY
        # --------------------------------------------

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )


        for fold, group in history.groupby(
            "fold"
        ):

            group = group.sort_values(
                "training_step"
            )

            ax.plot(
                group[
                    "training_step"
                ],
                (
                    group[
                        "val_balanced_accuracy"
                    ]
                    * 100
                ),
                marker="o",
                label=f"Fold {fold}",
            )


        ax.axhline(
            90,
            linestyle="--",
            linewidth=1,
            label="90% reference",
        )


        ax.set_xlabel(
            "Training Step"
        )

        ax.set_ylabel(
            "Validation Balanced Accuracy (%)"
        )

        ax.set_title(
            "V3 ResNet18 Validation Performance"
        )

        ax.set_ylim(
            0,
            105
        )

        ax.grid(
            alpha=0.25
        )

        ax.legend()


        fig.tight_layout()

        fig.savefig(
            BA_PNG,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


# ============================================================
# TEXT SUMMARY
# ============================================================

summary_lines = [

    "=" * 90,

    "FINAL PRIMARY MODEL RESULTS",

    "=" * 90,

    "",

    "Primary model:",
    "V3 ResNet18",

    "",

    "Evaluation scope:",
    (
        "Leakage-safe group-aware out-of-fold "
        "development evaluation."
    ),

    "",

    "IMPORTANT:",
    (
        "These V3 values are NOT external validation "
        "and must not be described as such."
    ),

    "",

    f"OOF cases:            {len(oof)}",

    f"Decision threshold:   {threshold:.17f}",

    "",

    f"Accuracy:             {accuracy * 100:.3f}%",

    (
        f"Balanced Accuracy:    "
        f"{balanced_accuracy * 100:.3f}%"
    ),

    f"Sensitivity:          {sensitivity * 100:.3f}%",

    f"Specificity:          {specificity * 100:.3f}%",

    f"Precision:            {precision * 100:.3f}%",

    f"F1 Score:             {f1 * 100:.3f}%",

    f"ROC-AUC:              {roc_auc * 100:.3f}%",

    f"PR-AUC:               {pr_auc * 100:.3f}%",

    f"MCC:                  {mcc:.6f}",

    "",

    (
        f"VeryMild sensitivity: "
        f"{very_mild_sensitivity * 100:.3f}%"
    ),

    (
        f"Mild sensitivity:     "
        f"{mild_sensitivity * 100:.3f}%"
    ),

    (
        f"Normal specificity:   "
        f"{normal_specificity * 100:.3f}%"
    ),

    "",

    (
        f"TN / FP / FN / TP:    "
        f"{tn} / {fp} / {fn} / {tp}"
    ),

    "",

    "Scientific interpretation:",
    (
        "V3 is the primary project model because it "
        "substantially improved internal group-aware "
        "performance over the classical ML stages."
    ),

    (
        "The separate OASIS experiment should be reported "
        "as a cross-domain generalization analysis and "
        "not merged with these primary internal metrics."
    ),

    "",

    "Research use only.",
    "Not a clinical diagnostic system.",
]


SUMMARY_TXT.write_text(
    "\n".join(
        summary_lines
    ),
    encoding="utf-8",
)


# ============================================================
# TERMINAL OUTPUT
# ============================================================

print()
print("=" * 90)
print("FINAL V3 PRIMARY RESULTS")
print("=" * 90)

print()

print(
    f"Accuracy:             "
    f"{accuracy * 100:.3f}%"
)

print(
    f"Balanced Accuracy:    "
    f"{balanced_accuracy * 100:.3f}%"
)

print(
    f"Sensitivity:          "
    f"{sensitivity * 100:.3f}%"
)

print(
    f"Specificity:          "
    f"{specificity * 100:.3f}%"
)

print(
    f"Precision:            "
    f"{precision * 100:.3f}%"
)

print(
    f"F1:                   "
    f"{f1 * 100:.3f}%"
)

print(
    f"ROC-AUC:              "
    f"{roc_auc * 100:.3f}%"
)

print(
    f"PR-AUC:               "
    f"{pr_auc * 100:.3f}%"
)

print(
    f"MCC:                  "
    f"{mcc:.6f}"
)

print()

print(
    f"VeryMild sensitivity: "
    f"{very_mild_sensitivity * 100:.3f}%"
)

print(
    f"Mild sensitivity:     "
    f"{mild_sensitivity * 100:.3f}%"
)

print(
    f"Normal specificity:   "
    f"{normal_specificity * 100:.3f}%"
)

print()

print(
    f"TN/FP/FN/TP: "
    f"{tn}/{fp}/{fn}/{tp}"
)

print()

print("=" * 90)
print("PACKAGE COMPLETE")
print("=" * 90)

print()

print("Saved to:")
print(OUTPUT_DIR)

print()

for file in [
    METRICS_CSV,
    METRICS_JSON,
    COMPARISON_CSV,
    CM_PNG,
    ROC_PNG,
    PR_PNG,
    COMPARE_PNG,
    LOSS_PNG,
    BA_PNG,
    SUMMARY_TXT,
]:

    if file.exists():
        print(file.name)

print()

print(
    "NO MODEL TRAINING OR TUNING WAS PERFORMED."
)