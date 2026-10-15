from pathlib import Path
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

try:
    from scipy.stats import binomtest
    SCIPY_BINOMTEST_AVAILABLE = True
except Exception:
    SCIPY_BINOMTEST_AVAILABLE = False


# ============================================================
# FINAL V2 TEST EVALUATION
# Alzheimer MRI Early Detection
#
# PURPOSE:
# Compare frozen V1 and frozen V2 on the SAME existing test set.
#
# IMPORTANT:
# - DOES NOT retrain either model.
# - DOES NOT rescale/refit.
# - DOES NOT tune thresholds.
# - DOES NOT modify V1 or V2.
# - Loads test.csv for evaluation only.
#
# SCIENTIFIC NOTE:
# This is a locked V2 comparison on the project's existing test set.
# The V2 scripts did not load this test set before V2 was frozen.
# However, the project had already evaluated V1 on this same test set
# earlier in development, so this should not be described as a brand-new
# pristine external test or independent external validation.
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
SPLIT_DIR = RESULTS_DIR / "GROUP_AWARE_SPLIT"

V1_DIR = RESULTS_DIR / "FINAL_MODEL"
V2_DIR = RESULTS_DIR / "FINAL_MODEL_V2"

OUTPUT_DIR = RESULTS_DIR / "FINAL_V2_TEST"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_FILE = SPLIT_DIR / "test.csv"

V1_MODEL_FILE = V1_DIR / "final_model.joblib"
V1_CONFIG_FILE = V1_DIR / "final_model_config.json"
V1_FEATURE_FILE = V1_DIR / "final_features.txt"

V2_MODEL_FILE = V2_DIR / "final_v2_model.joblib"
V2_CONFIG_FILE = V2_DIR / "final_v2_model_config.json"
V2_FEATURE_FILE = V2_DIR / "final_v2_features.txt"
V2_MANIFEST_FILE = V2_DIR / "freeze_manifest_v2.json"

PER_CASE_CSV = OUTPUT_DIR / "final_v1_v2_test_predictions.csv"
COMPARISON_CSV = OUTPUT_DIR / "final_v1_v2_metrics_comparison.csv"
REPORT_JSON = OUTPUT_DIR / "final_v2_test_report.json"
REPORT_TXT = OUTPUT_DIR / "final_v2_test_report.txt"


# ============================================================
# 2. EXPECTED LOCKED V2
# ============================================================

EXPECTED_V2 = {
    "kernel": "rbf",
    "C": 50.0,
    "gamma": 0.06,
    "class_weight": "balanced",
}

EXPECTED_V2_THRESHOLD = 0.085553268373468


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


def detect_original_class_column(df: pd.DataFrame):
    candidates = [
        "original_class",
        "original_label",
        "source_class",
        "class_name",
        "class",
        "label_name",
    ]

    lower_map = {c.lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def detect_identifier_columns(df: pd.DataFrame):
    candidates = [
        "filename",
        "file_name",
        "image_name",
        "image",
        "filepath",
        "file_path",
        "image_path",
        "path",
        "leakage_group",
        "original_class",
    ]

    lower_map = {c.lower(): c for c in df.columns}

    found = []
    for candidate in candidates:
        if candidate.lower() in lower_map:
            actual = lower_map[candidate.lower()]
            if actual not in found:
                found.append(actual)

    return found


def validate_test(df, features):
    if "target" not in df.columns:
        raise ValueError("test.csv is missing required column 'target'.")

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(
            f"test.csv is missing required feature columns: {missing}"
        )

    X = df[features].to_numpy(dtype=float)

    nan_count = int(np.isnan(X).sum())
    inf_count = int(np.isinf(X).sum())

    if nan_count != 0 or inf_count != 0:
        raise ValueError(
            f"Invalid test features: NaN={nan_count}, Inf={inf_count}"
        )

    targets = set(pd.unique(df["target"]))
    if not targets.issubset({0, 1}):
        raise ValueError(
            f"Target must be binary 0/1. Found: {targets}"
        )


def evaluate_scores(y_true, scores, threshold):
    pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    return {
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
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(
            average_precision_score(y_true, scores)
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }, pred


def subgroup_metrics(
    df,
    pred,
    original_class_col,
):
    out = {
        "verymild_sensitivity": None,
        "mild_sensitivity": None,
        "nondemented_specificity": None,
        "verymild_n": 0,
        "mild_n": 0,
        "nondemented_n": 0,
    }

    if original_class_col is None:
        return out

    classes = (
        df[original_class_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    def mask_for(names):
        return classes.isin(
            {name.lower() for name in names}
        ).to_numpy()

    verymild = mask_for([
        "VeryMildDemented",
        "Very Mild Demented",
        "Very_Mild_Demented",
    ])

    mild = mask_for([
        "MildDemented",
        "Mild Demented",
        "Mild_Demented",
    ])

    nondemented = mask_for([
        "NonDemented",
        "Non Demented",
        "Non_Demented",
        "Normal",
    ])

    out["verymild_n"] = int(verymild.sum())
    out["mild_n"] = int(mild.sum())
    out["nondemented_n"] = int(nondemented.sum())

    if verymild.any():
        out["verymild_sensitivity"] = float(
            np.mean(pred[verymild] == 1)
        )

    if mild.any():
        out["mild_sensitivity"] = float(
            np.mean(pred[mild] == 1)
        )

    if nondemented.any():
        out["nondemented_specificity"] = float(
            np.mean(pred[nondemented] == 0)
        )

    return out


def verify_v2_manifest():
    if not V2_MANIFEST_FILE.exists():
        return {
            "manifest_found": False,
            "model_hash_match": None,
        }

    with open(V2_MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest_hashes = manifest.get("sha256", {})
    expected_hash = manifest_hashes.get(V2_MODEL_FILE.name)

    actual_hash = sha256_file(V2_MODEL_FILE)

    hash_match = (
        expected_hash is not None
        and expected_hash == actual_hash
    )

    if expected_hash is not None and not hash_match:
        raise RuntimeError(
            "Frozen V2 model SHA256 does not match freeze manifest."
        )

    return {
        "manifest_found": True,
        "model_hash_match": hash_match,
        "actual_model_sha256": actual_hash,
        "manifest_model_sha256": expected_hash,
    }


def exact_mcnemar(y_true, pred_v1, pred_v2):
    correct_v1 = pred_v1 == y_true
    correct_v2 = pred_v2 == y_true

    # b: V1 correct, V2 wrong
    b = int(np.sum(correct_v1 & ~correct_v2))

    # c: V1 wrong, V2 correct
    c = int(np.sum(~correct_v1 & correct_v2))

    discordant = b + c

    if discordant == 0:
        p_value = 1.0
    elif SCIPY_BINOMTEST_AVAILABLE:
        p_value = float(
            binomtest(
                k=min(b, c),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
    else:
        p_value = None

    return {
        "v1_correct_v2_wrong": b,
        "v1_wrong_v2_correct": c,
        "discordant_pairs": discordant,
        "exact_two_sided_p_value": p_value,
        "scipy_binomtest_available": SCIPY_BINOMTEST_AVAILABLE,
    }


# ============================================================
# 4. HEADER
# ============================================================

print()
print("=" * 96)
print("FINAL LOCKED V2 TEST EVALUATION")
print("=" * 96)
print()
print("EVALUATION POLICY")
print("-----------------")
print("V1: frozen")
print("V2: frozen before this run")
print("No model fitting")
print("No threshold tuning")
print("No feature selection")
print("test.csv: loaded now for locked comparison")
print()


# ============================================================
# 5. VERIFY FILES
# ============================================================

required_files = [
    TEST_FILE,
    V1_MODEL_FILE,
    V1_CONFIG_FILE,
    V1_FEATURE_FILE,
    V2_MODEL_FILE,
    V2_CONFIG_FILE,
    V2_FEATURE_FILE,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(path)

manifest_check = verify_v2_manifest()


# ============================================================
# 6. LOAD CONFIGS / FEATURES
# ============================================================

with open(V1_CONFIG_FILE, "r", encoding="utf-8") as f:
    v1_config = json.load(f)

with open(V2_CONFIG_FILE, "r", encoding="utf-8") as f:
    v2_config = json.load(f)

v1_features = load_features(V1_FEATURE_FILE)
v2_features = load_features(V2_FEATURE_FILE)

if v1_features != v2_features:
    raise RuntimeError(
        "V1 and V2 feature lists/order do not match exactly."
    )

features = v1_features

if len(features) != 30:
    raise RuntimeError(
        f"Expected exactly 30 features, found {len(features)}."
    )

v1_threshold = float(
    v1_config["selected_threshold"]
)

v2_threshold = float(
    v2_config["selected_threshold"]
)

if not np.isclose(
    v2_threshold,
    EXPECTED_V2_THRESHOLD,
    atol=1e-12,
    rtol=0.0,
):
    raise RuntimeError(
        "Frozen V2 threshold does not match the locked "
        "validation-selected threshold."
    )

v2_params = v2_config.get("model_parameters", {})

if v2_params.get("kernel") != EXPECTED_V2["kernel"]:
    raise RuntimeError("Frozen V2 kernel mismatch.")

if not np.isclose(
    float(v2_params.get("C")),
    EXPECTED_V2["C"],
    atol=1e-15,
    rtol=0.0,
):
    raise RuntimeError("Frozen V2 C mismatch.")

if not np.isclose(
    float(v2_params.get("gamma")),
    EXPECTED_V2["gamma"],
    atol=1e-15,
    rtol=0.0,
):
    raise RuntimeError("Frozen V2 gamma mismatch.")

if v2_params.get("class_weight") != EXPECTED_V2["class_weight"]:
    raise RuntimeError("Frozen V2 class_weight mismatch.")

print("Frozen configuration verification: PASS")
print(f"V1 threshold = {v1_threshold:.15f}")
print(f"V2 threshold = {v2_threshold:.15f}")
print(f"Feature count = {len(features)}")


# ============================================================
# 7. LOAD FROZEN MODELS
# ============================================================

v1_model = joblib.load(V1_MODEL_FILE)
v2_model = joblib.load(V2_MODEL_FILE)

if not hasattr(v1_model, "decision_function"):
    raise TypeError("Frozen V1 model lacks decision_function.")

if not hasattr(v2_model, "decision_function"):
    raise TypeError("Frozen V2 model lacks decision_function.")


# ============================================================
# 8. LOAD TEST SET
# ============================================================

print()
print("=" * 96)
print("LOADING EXISTING TEST SET")
print("=" * 96)

test_df = pd.read_csv(TEST_FILE)

validate_test(
    test_df,
    features,
)

print(f"Test rows: {len(test_df)}")

X_test = test_df[features]
y_test = (
    test_df["target"]
    .astype(int)
    .to_numpy()
)

original_class_col = detect_original_class_column(
    test_df
)

print(
    f"Original class column: {original_class_col}"
)


# ============================================================
# 9. LOCKED PREDICTIONS
# ============================================================

v1_scores = v1_model.decision_function(
    X_test
)

v2_scores = v2_model.decision_function(
    X_test
)

v1_metrics, v1_pred = evaluate_scores(
    y_test,
    v1_scores,
    v1_threshold,
)

v2_metrics, v2_pred = evaluate_scores(
    y_test,
    v2_scores,
    v2_threshold,
)

v1_subgroups = subgroup_metrics(
    test_df,
    v1_pred,
    original_class_col,
)

v2_subgroups = subgroup_metrics(
    test_df,
    v2_pred,
    original_class_col,
)

v1_all = {
    **v1_metrics,
    **v1_subgroups,
}

v2_all = {
    **v2_metrics,
    **v2_subgroups,
}


# ============================================================
# 10. PAIRED MCNEMAR
# ============================================================

mcnemar = exact_mcnemar(
    y_test,
    v1_pred,
    v2_pred,
)


# ============================================================
# 11. PER-CASE OUTPUT
# ============================================================

id_cols = detect_identifier_columns(
    test_df
)

per_case = pd.DataFrame(
    index=np.arange(len(test_df))
)

per_case["row_index"] = np.arange(
    len(test_df)
)

for col in id_cols:
    per_case[col] = test_df[col].values

per_case["target"] = y_test

per_case["v1_score"] = v1_scores
per_case["v1_threshold"] = v1_threshold
per_case["v1_prediction"] = v1_pred
per_case["v1_correct"] = (
    v1_pred == y_test
).astype(int)

per_case["v2_score"] = v2_scores
per_case["v2_threshold"] = v2_threshold
per_case["v2_prediction"] = v2_pred
per_case["v2_correct"] = (
    v2_pred == y_test
).astype(int)

per_case["prediction_changed"] = (
    v1_pred != v2_pred
).astype(int)

per_case["v1_wrong_v2_correct"] = (
    (v1_pred != y_test)
    &
    (v2_pred == y_test)
).astype(int)

per_case["v1_correct_v2_wrong"] = (
    (v1_pred == y_test)
    &
    (v2_pred != y_test)
).astype(int)

per_case.to_csv(
    PER_CASE_CSV,
    index=False,
)


# ============================================================
# 12. COMPARISON TABLE
# ============================================================

metric_order = [
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "roc_auc",
    "pr_auc",
    "verymild_sensitivity",
    "mild_sensitivity",
    "nondemented_specificity",
    "TN",
    "FP",
    "FN",
    "TP",
]

comparison_rows = []

for metric in metric_order:
    v1_value = v1_all.get(metric)
    v2_value = v2_all.get(metric)

    delta = None

    if (
        v1_value is not None
        and v2_value is not None
    ):
        delta = float(v2_value) - float(v1_value)

    comparison_rows.append({
        "metric": metric,
        "V1": v1_value,
        "V2": v2_value,
        "V2_minus_V1": delta,
    })

comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df.to_csv(
    COMPARISON_CSV,
    index=False,
)


# ============================================================
# 13. DECISION SUMMARY
# ============================================================

ba_delta = (
    v2_metrics["balanced_accuracy"]
    -
    v1_metrics["balanced_accuracy"]
)

accuracy_delta = (
    v2_metrics["accuracy"]
    -
    v1_metrics["accuracy"]
)

roc_delta = (
    v2_metrics["roc_auc"]
    -
    v1_metrics["roc_auc"]
)

vm_delta = None

if (
    v1_subgroups["verymild_sensitivity"] is not None
    and
    v2_subgroups["verymild_sensitivity"] is not None
):
    vm_delta = (
        v2_subgroups["verymild_sensitivity"]
        -
        v1_subgroups["verymild_sensitivity"]
    )

if ba_delta > 0:
    primary_outcome = (
        "V2 improved test balanced accuracy versus V1."
    )
elif ba_delta < 0:
    primary_outcome = (
        "V2 reduced test balanced accuracy versus V1."
    )
else:
    primary_outcome = (
        "V2 and V1 had identical test balanced accuracy."
    )


# ============================================================
# 14. JSON REPORT
# ============================================================

report = {
    "status": "LOCKED_V2_TEST_EVALUATED",
    "scientific_scope": (
        "Locked V2 comparison on the project's existing held-out "
        "test split. V2 was frozen before this evaluation. "
        "However, V1 had already been evaluated on this same test "
        "set earlier in the project, so this is not a newly pristine "
        "independent external validation."
    ),
    "environment": {
        "python": sys.version,
        "scikit_learn": sklearn.__version__,
    },
    "test_rows": int(len(test_df)),
    "feature_count": int(len(features)),
    "original_class_column": original_class_col,
    "V1": {
        "threshold": v1_threshold,
        "metrics": v1_all,
        "model_sha256": sha256_file(
            V1_MODEL_FILE
        ),
    },
    "V2": {
        "threshold": v2_threshold,
        "model_parameters": EXPECTED_V2,
        "metrics": v2_all,
        "model_sha256": sha256_file(
            V2_MODEL_FILE
        ),
        "freeze_manifest_check": manifest_check,
    },
    "deltas_V2_minus_V1": {
        "accuracy": accuracy_delta,
        "balanced_accuracy": ba_delta,
        "roc_auc": roc_delta,
        "verymild_sensitivity": vm_delta,
    },
    "paired_mcnemar_exact": mcnemar,
    "primary_outcome": primary_outcome,
}

with open(
    REPORT_JSON,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        report,
        f,
        indent=4,
    )


# ============================================================
# 15. TERMINAL OUTPUT
# ============================================================

print()
print("=" * 96)
print("FINAL TEST RESULTS")
print("=" * 96)

print()
print("V1")
print("-" * 40)
print(
    f"Accuracy:          "
    f"{v1_metrics['accuracy']:.6f}"
)
print(
    f"Balanced Accuracy: "
    f"{v1_metrics['balanced_accuracy']:.6f}"
)
print(
    f"Sensitivity:       "
    f"{v1_metrics['sensitivity']:.6f}"
)
print(
    f"Specificity:       "
    f"{v1_metrics['specificity']:.6f}"
)
print(
    f"Precision:         "
    f"{v1_metrics['precision']:.6f}"
)
print(
    f"F1:                "
    f"{v1_metrics['f1']:.6f}"
)
print(
    f"ROC-AUC:           "
    f"{v1_metrics['roc_auc']:.6f}"
)
print(
    f"PR-AUC:            "
    f"{v1_metrics['pr_auc']:.6f}"
)
print(
    f"TN/FP/FN/TP:       "
    f"{v1_metrics['TN']}/"
    f"{v1_metrics['FP']}/"
    f"{v1_metrics['FN']}/"
    f"{v1_metrics['TP']}"
)
print(
    f"VeryMild Sens.:    "
    f"{v1_subgroups['verymild_sensitivity']}"
)
print(
    f"Mild Sens.:        "
    f"{v1_subgroups['mild_sensitivity']}"
)
print(
    f"NonDemented Spec.: "
    f"{v1_subgroups['nondemented_specificity']}"
)

print()
print("V2")
print("-" * 40)
print(
    f"Accuracy:          "
    f"{v2_metrics['accuracy']:.6f}"
)
print(
    f"Balanced Accuracy: "
    f"{v2_metrics['balanced_accuracy']:.6f}"
)
print(
    f"Sensitivity:       "
    f"{v2_metrics['sensitivity']:.6f}"
)
print(
    f"Specificity:       "
    f"{v2_metrics['specificity']:.6f}"
)
print(
    f"Precision:         "
    f"{v2_metrics['precision']:.6f}"
)
print(
    f"F1:                "
    f"{v2_metrics['f1']:.6f}"
)
print(
    f"ROC-AUC:           "
    f"{v2_metrics['roc_auc']:.6f}"
)
print(
    f"PR-AUC:            "
    f"{v2_metrics['pr_auc']:.6f}"
)
print(
    f"TN/FP/FN/TP:       "
    f"{v2_metrics['TN']}/"
    f"{v2_metrics['FP']}/"
    f"{v2_metrics['FN']}/"
    f"{v2_metrics['TP']}"
)
print(
    f"VeryMild Sens.:    "
    f"{v2_subgroups['verymild_sensitivity']}"
)
print(
    f"Mild Sens.:        "
    f"{v2_subgroups['mild_sensitivity']}"
)
print(
    f"NonDemented Spec.: "
    f"{v2_subgroups['nondemented_specificity']}"
)

print()
print("=" * 96)
print("V2 - V1 DELTAS")
print("=" * 96)
print(
    f"Accuracy delta:          "
    f"{accuracy_delta:+.6f}"
)
print(
    f"Balanced Accuracy delta: "
    f"{ba_delta:+.6f}"
)
print(
    f"ROC-AUC delta:           "
    f"{roc_delta:+.6f}"
)
print(
    f"VeryMild Sens. delta:    "
    f"{vm_delta}"
)

print()
print("=" * 96)
print("PAIRED MCNEMAR EXACT TEST")
print("=" * 96)
print(
    "V1 correct / V2 wrong: "
    f"{mcnemar['v1_correct_v2_wrong']}"
)
print(
    "V1 wrong / V2 correct: "
    f"{mcnemar['v1_wrong_v2_correct']}"
)
print(
    "Discordant pairs:       "
    f"{mcnemar['discordant_pairs']}"
)
print(
    "Exact two-sided p-value:"
    f" {mcnemar['exact_two_sided_p_value']}"
)

print()
print("=" * 96)
print("PRIMARY OUTCOME")
print("=" * 96)
print(primary_outcome)

print()
print("Saved:")
print(PER_CASE_CSV)
print(COMPARISON_CSV)
print(REPORT_JSON)
print(REPORT_TXT)


# ============================================================
# 16. TEXT REPORT
# ============================================================

text_lines = [
    "=" * 96,
    "FINAL V1 VS V2 TEST REPORT",
    "=" * 96,
    "",
    "SCIENTIFIC SCOPE",
    (
        "V2 was frozen before this test evaluation. "
        "This compares locked V1 and locked V2 on the same "
        "existing project test split. V1 had already been "
        "evaluated on this test set earlier, so this should "
        "not be described as newly pristine independent "
        "external validation."
    ),
    "",
    "V1",
    f"Threshold: {v1_threshold:.15f}",
    f"Accuracy: {v1_metrics['accuracy']:.12f}",
    (
        "Balanced Accuracy: "
        f"{v1_metrics['balanced_accuracy']:.12f}"
    ),
    f"Sensitivity: {v1_metrics['sensitivity']:.12f}",
    f"Specificity: {v1_metrics['specificity']:.12f}",
    f"Precision: {v1_metrics['precision']:.12f}",
    f"F1: {v1_metrics['f1']:.12f}",
    f"ROC-AUC: {v1_metrics['roc_auc']:.12f}",
    f"PR-AUC: {v1_metrics['pr_auc']:.12f}",
    (
        "TN/FP/FN/TP: "
        f"{v1_metrics['TN']}/"
        f"{v1_metrics['FP']}/"
        f"{v1_metrics['FN']}/"
        f"{v1_metrics['TP']}"
    ),
    (
        "VeryMild sensitivity: "
        f"{v1_subgroups['verymild_sensitivity']}"
    ),
    (
        "Mild sensitivity: "
        f"{v1_subgroups['mild_sensitivity']}"
    ),
    (
        "NonDemented specificity: "
        f"{v1_subgroups['nondemented_specificity']}"
    ),
    "",
    "V2",
    f"Threshold: {v2_threshold:.15f}",
    f"C: {EXPECTED_V2['C']}",
    f"gamma: {EXPECTED_V2['gamma']}",
    f"Accuracy: {v2_metrics['accuracy']:.12f}",
    (
        "Balanced Accuracy: "
        f"{v2_metrics['balanced_accuracy']:.12f}"
    ),
    f"Sensitivity: {v2_metrics['sensitivity']:.12f}",
    f"Specificity: {v2_metrics['specificity']:.12f}",
    f"Precision: {v2_metrics['precision']:.12f}",
    f"F1: {v2_metrics['f1']:.12f}",
    f"ROC-AUC: {v2_metrics['roc_auc']:.12f}",
    f"PR-AUC: {v2_metrics['pr_auc']:.12f}",
    (
        "TN/FP/FN/TP: "
        f"{v2_metrics['TN']}/"
        f"{v2_metrics['FP']}/"
        f"{v2_metrics['FN']}/"
        f"{v2_metrics['TP']}"
    ),
    (
        "VeryMild sensitivity: "
        f"{v2_subgroups['verymild_sensitivity']}"
    ),
    (
        "Mild sensitivity: "
        f"{v2_subgroups['mild_sensitivity']}"
    ),
    (
        "NonDemented specificity: "
        f"{v2_subgroups['nondemented_specificity']}"
    ),
    "",
    "V2 - V1",
    f"Accuracy delta: {accuracy_delta:+.12f}",
    (
        "Balanced Accuracy delta: "
        f"{ba_delta:+.12f}"
    ),
    f"ROC-AUC delta: {roc_delta:+.12f}",
    f"VeryMild sensitivity delta: {vm_delta}",
    "",
    "PAIRED MCNEMAR EXACT",
    (
        "V1 correct / V2 wrong: "
        f"{mcnemar['v1_correct_v2_wrong']}"
    ),
    (
        "V1 wrong / V2 correct: "
        f"{mcnemar['v1_wrong_v2_correct']}"
    ),
    (
        "Discordant pairs: "
        f"{mcnemar['discordant_pairs']}"
    ),
    (
        "Exact two-sided p-value: "
        f"{mcnemar['exact_two_sided_p_value']}"
    ),
    "",
    "PRIMARY OUTCOME",
    primary_outcome,
    "",
    "=" * 96,
]

with open(
    REPORT_TXT,
    "w",
    encoding="utf-8",
) as f:
    f.write("\n".join(text_lines))
