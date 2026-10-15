from pathlib import Path
import json
import platform
import sys
import time
from itertools import product

import numpy as np
import pandas as pd
import sklearn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ============================================================
# V2 SVM GROUP-AWARE CV TUNING
# Alzheimer MRI Early Detection
#
# IMPORTANT:
# - TRAINING SPLIT ONLY.
# - validation.csv is NOT loaded.
# - test.csv is NOT loaded.
# - Group-aware CV uses leakage_group.
# - Scaling happens INSIDE each CV fold.
# - No threshold optimization here.
# - No final V2 model is frozen here.
# - Script checkpoints after every hyperparameter configuration.
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
SPLIT_DIR = RESULTS_DIR / "GROUP_AWARE_SPLIT"
FINAL_MODEL_DIR = RESULTS_DIR / "FINAL_MODEL"
OUTPUT_DIR = RESULTS_DIR / "V2_SVM_GROUP_CV"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = SPLIT_DIR / "train.csv"
FEATURE_FILE = FINAL_MODEL_DIR / "final_features.txt"

FOLD_RESULTS_CSV = OUTPUT_DIR / "v2_svm_group_cv_fold_results.csv"
SUMMARY_CSV = OUTPUT_DIR / "v2_svm_group_cv_ranked_summary.csv"
REPORT_JSON = OUTPUT_DIR / "v2_svm_group_cv_report.json"
BEST_PARAMS_JSON = OUTPUT_DIR / "best_hyperparameters.json"
PROGRESS_JSON = OUTPUT_DIR / "progress.json"


# ============================================================
# 2. CONFIGURATION
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 3

C_VALUES = [20, 25, 30, 40, 50, 60]
GAMMA_VALUES = [0.05, 0.06, 0.075, 0.09, 0.10]

CONFIGS = list(product(C_VALUES, GAMMA_VALUES))


# ============================================================
# 3. HELPERS
# ============================================================

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


def validate_training_dataframe(df: pd.DataFrame, features):
    required = {"target", "leakage_group"}
    missing_required = sorted(required - set(df.columns))
    if missing_required:
        raise ValueError(
            f"Training data missing required columns: {missing_required}"
        )

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(
            f"Training data missing frozen features: {missing_features}"
        )

    X = df[features].to_numpy(dtype=float)

    nan_count = int(np.isnan(X).sum())
    inf_count = int(np.isinf(X).sum())

    if nan_count != 0 or inf_count != 0:
        raise ValueError(
            f"Invalid feature matrix: NaN={nan_count}, Inf={inf_count}"
        )

    targets = set(pd.unique(df["target"]))
    if not targets.issubset({0, 1}):
        raise ValueError(
            f"Target must be binary 0/1. Found: {targets}"
        )


def compute_metrics(y_true, scores):
    """
    SVC decision_function default threshold = 0.
    """
    pred = (scores >= 0.0).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, pred, labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, pred)
        ),
        "sensitivity": float(
            recall_score(
                y_true,
                pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "specificity": float(specificity),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(
            average_precision_score(y_true, scores)
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def compute_subgroups(
    fold_df: pd.DataFrame,
    scores,
    original_class_col,
):
    result = {
        "verymild_sensitivity": np.nan,
        "mild_sensitivity": np.nan,
        "nondemented_specificity": np.nan,
        "verymild_n": 0,
        "mild_n": 0,
        "nondemented_n": 0,
    }

    if original_class_col is None:
        return result

    pred = (scores >= 0.0).astype(int)

    classes = (
        fold_df[original_class_col]
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

    result["verymild_n"] = int(verymild.sum())
    result["mild_n"] = int(mild.sum())
    result["nondemented_n"] = int(nondemented.sum())

    if verymild.any():
        result["verymild_sensitivity"] = float(
            np.mean(pred[verymild] == 1)
        )

    if mild.any():
        result["mild_sensitivity"] = float(
            np.mean(pred[mild] == 1)
        )

    if nondemented.any():
        result["nondemented_specificity"] = float(
            np.mean(pred[nondemented] == 0)
        )

    return result


def mean_or_nan(series):
    series = pd.to_numeric(series, errors="coerce")
    if series.notna().sum() == 0:
        return np.nan
    return float(series.mean())


def std_or_nan(series):
    series = pd.to_numeric(series, errors="coerce")
    if series.notna().sum() <= 1:
        return np.nan
    return float(series.std(ddof=1))


def save_progress(completed_count, total_count, last_config):
    payload = {
        "completed_configurations": int(completed_count),
        "total_configurations": int(total_count),
        "last_completed": last_config,
        "note": (
            "Safe checkpoint. Re-running the script will skip "
            "configurations already present in the fold-results CSV."
        ),
    }
    with open(PROGRESS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


# ============================================================
# 4. LOAD TRAINING DATA ONLY
# ============================================================

print()
print("=" * 92)
print("V2 SVM GROUP-AWARE CV TUNING - TRAINING SPLIT ONLY")
print("=" * 92)
print()
print("DATA POLICY")
print("-----------")
print("train.csv      : LOADED")
print("validation.csv : NOT LOADED")
print("test.csv       : NOT LOADED")
print()

if not TRAIN_FILE.exists():
    raise FileNotFoundError(TRAIN_FILE)

if not FEATURE_FILE.exists():
    raise FileNotFoundError(FEATURE_FILE)

features = load_features(FEATURE_FILE)

if len(features) != 30:
    raise ValueError(
        f"Expected 30 frozen V1 features, found {len(features)}."
    )

train_df = pd.read_csv(TRAIN_FILE)
validate_training_dataframe(train_df, features)

original_class_col = detect_original_class_column(train_df)

X = train_df[features]
y = train_df["target"].astype(int).to_numpy()
groups = train_df["leakage_group"].astype(str).to_numpy()

print(f"Training rows:        {len(train_df)}")
print(f"Frozen features:      {len(features)}")
print(f"Unique leakage groups:{len(np.unique(groups))}")
print(f"Original class column:{original_class_col}")
print(f"CV folds:             {N_SPLITS}")
print(f"Configurations:       {len(CONFIGS)}")
print()


# ============================================================
# 5. BUILD FIXED GROUP-AWARE FOLDS
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

splits = list(cv.split(X, y, groups=groups))

print("=" * 92)
print("CV FOLD AUDIT")
print("=" * 92)

for fold_no, (fit_idx, eval_idx) in enumerate(splits, start=1):

    fit_groups = set(groups[fit_idx])
    eval_groups = set(groups[eval_idx])

    overlap = fit_groups & eval_groups

    if overlap:
        raise RuntimeError(
            f"Fold {fold_no}: leakage-group overlap detected."
        )

    fit_y = y[fit_idx]
    eval_y = y[eval_idx]

    print(
        f"Fold {fold_no}: "
        f"fit={len(fit_idx)} "
        f"eval={len(eval_idx)} "
        f"group_overlap=0 "
        f"fit_early={int((fit_y == 1).sum())} "
        f"fit_normal={int((fit_y == 0).sum())} "
        f"eval_early={int((eval_y == 1).sum())} "
        f"eval_normal={int((eval_y == 0).sum())}"
    )

print()


# ============================================================
# 6. RESUME SUPPORT
# ============================================================

existing_rows = []

if FOLD_RESULTS_CSV.exists():
    existing_df = pd.read_csv(FOLD_RESULTS_CSV)

    required_resume_cols = {"C", "gamma", "fold"}
    if required_resume_cols.issubset(existing_df.columns):
        existing_rows = existing_df.to_dict(orient="records")
        print(
            f"Resume mode: found {len(existing_rows)} existing fold results."
        )
    else:
        print(
            "Existing fold-results CSV has incompatible columns. "
            "Starting fresh."
        )
        existing_rows = []

completed_pairs = {
    (float(row["C"]), float(row["gamma"]), int(row["fold"]))
    for row in existing_rows
    if (
        pd.notna(row.get("C"))
        and pd.notna(row.get("gamma"))
        and pd.notna(row.get("fold"))
    )
}

all_rows = list(existing_rows)


# ============================================================
# 7. GRID SEARCH
# ============================================================

print()
print("=" * 92)
print("RUNNING TRAINING-ONLY GROUP-AWARE GRID")
print("=" * 92)

total_configs = len(CONFIGS)

for config_index, (C_value, gamma_value) in enumerate(
    CONFIGS,
    start=1,
):

    config_key_rows = [
        (
            float(C_value),
            float(gamma_value),
            fold_no,
        )
        for fold_no in range(1, N_SPLITS + 1)
    ]

    if all(key in completed_pairs for key in config_key_rows):
        print(
            f"[{config_index}/{total_configs}] "
            f"C={C_value}, gamma={gamma_value} -> already complete, skipping."
        )
        continue

    print()
    print(
        f"[{config_index}/{total_configs}] "
        f"C={C_value}, gamma={gamma_value}"
    )

    config_start = time.perf_counter()

    for fold_no, (fit_idx, eval_idx) in enumerate(
        splits,
        start=1,
    ):

        pair_key = (
            float(C_value),
            float(gamma_value),
            fold_no,
        )

        if pair_key in completed_pairs:
            print(f"  Fold {fold_no}: already complete, skipping.")
            continue

        fold_start = time.perf_counter()

        model = Pipeline([
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    C=C_value,
                    gamma=gamma_value,
                    class_weight="balanced",
                    probability=False,
                    cache_size=4000,
                ),
            ),
        ])

        X_fit = X.iloc[fit_idx]
        y_fit = y[fit_idx]

        X_eval = X.iloc[eval_idx]
        y_eval = y[eval_idx]

        model.fit(X_fit, y_fit)

        scores = model.decision_function(X_eval)

        metrics = compute_metrics(y_eval, scores)

        fold_df = train_df.iloc[eval_idx].reset_index(drop=True)

        subgroups = compute_subgroups(
            fold_df,
            scores,
            original_class_col,
        )

        elapsed = time.perf_counter() - fold_start

        row = {
            "C": float(C_value),
            "gamma": float(gamma_value),
            "fold": int(fold_no),
            "fit_rows": int(len(fit_idx)),
            "eval_rows": int(len(eval_idx)),
            "fit_seconds": float(elapsed),
            **metrics,
            **subgroups,
        }

        all_rows.append(row)
        completed_pairs.add(pair_key)

        # checkpoint after every fold
        pd.DataFrame(all_rows).to_csv(
            FOLD_RESULTS_CSV,
            index=False,
        )

        print(
            f"  Fold {fold_no}: "
            f"BA={metrics['balanced_accuracy']:.6f} "
            f"VM={subgroups['verymild_sensitivity']:.6f} "
            f"ROC={metrics['roc_auc']:.6f} "
            f"time={elapsed:.1f}s"
        )

    config_elapsed = time.perf_counter() - config_start

    completed_configs = 0
    for c, g in CONFIGS:
        keys = [
            (float(c), float(g), f)
            for f in range(1, N_SPLITS + 1)
        ]
        if all(k in completed_pairs for k in keys):
            completed_configs += 1

    save_progress(
        completed_count=completed_configs,
        total_count=total_configs,
        last_config={
            "C": float(C_value),
            "gamma": float(gamma_value),
            "elapsed_seconds": float(config_elapsed),
        },
    )

    print(
        f"  Configuration complete in {config_elapsed:.1f}s "
        f"({completed_configs}/{total_configs} configs finished)"
    )


# ============================================================
# 8. BUILD RANKED SUMMARY
# ============================================================

fold_df = pd.DataFrame(all_rows)

# Keep only complete configurations with all folds present.
counts = (
    fold_df
    .groupby(["C", "gamma"])["fold"]
    .nunique()
    .reset_index(name="fold_count")
)

complete = counts[counts["fold_count"] == N_SPLITS][
    ["C", "gamma"]
]

complete_fold_df = fold_df.merge(
    complete,
    on=["C", "gamma"],
    how="inner",
)

summary_rows = []

for (C_value, gamma_value), grp in complete_fold_df.groupby(
    ["C", "gamma"],
    sort=False,
):
    summary_rows.append({
        "C": float(C_value),
        "gamma": float(gamma_value),
        "folds": int(grp["fold"].nunique()),

        "mean_balanced_accuracy": mean_or_nan(
            grp["balanced_accuracy"]
        ),
        "std_balanced_accuracy": std_or_nan(
            grp["balanced_accuracy"]
        ),

        "mean_accuracy": mean_or_nan(grp["accuracy"]),
        "std_accuracy": std_or_nan(grp["accuracy"]),

        "mean_sensitivity": mean_or_nan(grp["sensitivity"]),
        "mean_specificity": mean_or_nan(grp["specificity"]),

        "mean_roc_auc": mean_or_nan(grp["roc_auc"]),
        "std_roc_auc": std_or_nan(grp["roc_auc"]),

        "mean_pr_auc": mean_or_nan(grp["pr_auc"]),
        "std_pr_auc": std_or_nan(grp["pr_auc"]),

        "mean_verymild_sensitivity": mean_or_nan(
            grp["verymild_sensitivity"]
        ),
        "std_verymild_sensitivity": std_or_nan(
            grp["verymild_sensitivity"]
        ),

        "mean_mild_sensitivity": mean_or_nan(
            grp["mild_sensitivity"]
        ),

        "mean_nondemented_specificity": mean_or_nan(
            grp["nondemented_specificity"]
        ),

        "mean_fit_seconds_per_fold": mean_or_nan(
            grp["fit_seconds"]
        ),
        "total_fit_seconds": float(
            pd.to_numeric(
                grp["fit_seconds"],
                errors="coerce",
            ).sum()
        ),
    })

summary_df = pd.DataFrame(summary_rows)

if summary_df.empty:
    raise RuntimeError(
        "No complete hyperparameter configurations were available."
    )

summary_df["_vm_rank"] = (
    summary_df["mean_verymild_sensitivity"]
    .fillna(-1.0)
)

summary_df = summary_df.sort_values(
    by=[
        "mean_balanced_accuracy",
        "_vm_rank",
        "mean_roc_auc",
    ],
    ascending=[
        False,
        False,
        False,
    ],
).reset_index(drop=True)

summary_df.insert(
    0,
    "rank",
    np.arange(1, len(summary_df) + 1),
)

summary_df = summary_df.drop(columns=["_vm_rank"])

summary_df.to_csv(
    SUMMARY_CSV,
    index=False,
)


# ============================================================
# 9. BEST HYPERPARAMETERS
# ============================================================

best = summary_df.iloc[0]

best_params = {
    "status": "TRAINING_CV_SELECTED_ONLY",
    "C": float(best["C"]),
    "gamma": float(best["gamma"]),
    "kernel": "rbf",
    "class_weight": "balanced",
    "scaling": "StandardScaler inside each CV fold",
    "cv": {
        "type": "StratifiedGroupKFold",
        "n_splits": N_SPLITS,
        "shuffle": True,
        "random_state": RANDOM_STATE,
        "group_column": "leakage_group",
    },
    "ranking_rule": [
        "mean CV balanced accuracy",
        "mean VeryMildDemented sensitivity",
        "mean CV ROC-AUC",
    ],
    "mean_cv_balanced_accuracy": float(
        best["mean_balanced_accuracy"]
    ),
    "std_cv_balanced_accuracy": float(
        best["std_balanced_accuracy"]
    ),
    "mean_cv_verymild_sensitivity": (
        None
        if pd.isna(best["mean_verymild_sensitivity"])
        else float(best["mean_verymild_sensitivity"])
    ),
    "mean_cv_roc_auc": float(
        best["mean_roc_auc"]
    ),
    "important": (
        "Validation and test were not loaded. "
        "This selects hyperparameters only. "
        "Do not call this final V2 performance."
    ),
}

with open(BEST_PARAMS_JSON, "w", encoding="utf-8") as f:
    json.dump(best_params, f, indent=4)


# ============================================================
# 10. FULL REPORT
# ============================================================

report = {
    "status": "TRAINING_ONLY_DEVELOPMENT",
    "data_policy": {
        "train_csv_loaded": True,
        "validation_csv_loaded": False,
        "test_csv_loaded": False,
    },
    "training_rows": int(len(train_df)),
    "unique_leakage_groups": int(
        len(np.unique(groups))
    ),
    "feature_count": int(len(features)),
    "features": features,
    "original_class_column": original_class_col,
    "grid": {
        "C_values": C_VALUES,
        "gamma_values": GAMMA_VALUES,
        "configuration_count": len(CONFIGS),
    },
    "cv": {
        "type": "StratifiedGroupKFold",
        "n_splits": N_SPLITS,
        "shuffle": True,
        "random_state": RANDOM_STATE,
        "group_column": "leakage_group",
    },
    "ranking_rule": {
        "primary": "mean_balanced_accuracy",
        "secondary": "mean_verymild_sensitivity",
        "tertiary": "mean_roc_auc",
    },
    "environment": {
        "python": sys.version,
        "platform": platform.platform(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    },
    "best_hyperparameters": best_params,
    "summary": summary_df.to_dict(orient="records"),
}

with open(REPORT_JSON, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print()
print("=" * 92)
print("TRAINING-ONLY GROUP-AWARE CV RANKING")
print("=" * 92)

display_cols = [
    "rank",
    "C",
    "gamma",
    "mean_balanced_accuracy",
    "std_balanced_accuracy",
    "mean_verymild_sensitivity",
    "mean_mild_sensitivity",
    "mean_nondemented_specificity",
    "mean_roc_auc",
    "mean_pr_auc",
]

print(
    summary_df[display_cols].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print()
print("=" * 92)
print("SELECTED TRAINING-CV HYPERPARAMETERS")
print("=" * 92)
print(f"C     = {best_params['C']}")
print(f"gamma = {best_params['gamma']}")
print(
    f"Mean CV balanced accuracy = "
    f"{best_params['mean_cv_balanced_accuracy']:.6f}"
)
print(
    f"Mean CV VeryMild sensitivity = "
    f"{best_params['mean_cv_verymild_sensitivity']}"
)
print(
    f"Mean CV ROC-AUC = "
    f"{best_params['mean_cv_roc_auc']:.6f}"
)

print()
print("IMPORTANT:")
print("Validation and test remain untouched by this script.")
print("Do NOT freeze V2 yet.")
print()
print("Saved:")
print(FOLD_RESULTS_CSV)
print(SUMMARY_CSV)
print(BEST_PARAMS_JSON)
print(REPORT_JSON)
