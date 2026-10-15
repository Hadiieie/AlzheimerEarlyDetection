from pathlib import Path
import json
import hashlib

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# V3 CNN DATASET PREPARATION
# Alzheimer MRI Early Detection
#
# PURPOSE:
# Build a leakage-safe TRAINING-ONLY manifest for V3 deep learning.
#
# IMPORTANT:
# - Uses ONLY the existing train.csv.
# - Does NOT load validation.csv.
# - Does NOT load test.csv.
# - Uses leakage_group for 3-fold group-aware CV.
# - Verifies that processed MRI files exist and are readable.
# - Locks fold assignment before any CNN training.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
SPLIT_DIR = RESULTS_DIR / "GROUP_AWARE_SPLIT"
OUTPUT_DIR = RESULTS_DIR / "V3_CNN_DATASET"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = SPLIT_DIR / "train.csv"

MANIFEST_FILE = OUTPUT_DIR / "v3_training_cv_manifest.csv"
SUMMARY_JSON = OUTPUT_DIR / "v3_dataset_summary.json"
SUMMARY_TXT = OUTPUT_DIR / "v3_dataset_summary.txt"
MISSING_FILE = OUTPUT_DIR / "v3_missing_or_invalid_images.csv"

RANDOM_STATE = 42
N_SPLITS = 3


# ============================================================
# HELPERS
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


def resolve_image_column(df: pd.DataFrame):
    """
    Prefer processed_path because the existing project already
    standardized these images to 220x220 with CLAHE + median filter.
    Fall back to path only if processed_path is unavailable.
    """
    if "processed_path" in df.columns:
        return "processed_path"

    if "path" in df.columns:
        return "path"

    raise ValueError(
        "Neither 'processed_path' nor 'path' exists in train.csv."
    )


def verify_image(path_string):
    path = Path(str(path_string))

    if not path.exists():
        return False, "FILE_NOT_FOUND", None

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        return False, "UNREADABLE_IMAGE", None

    h, w = image.shape[:2]

    return True, "PASS", (int(h), int(w))


# ============================================================
# LOAD TRAINING SPLIT ONLY
# ============================================================

print()
print("=" * 92)
print("V3 CNN DATASET PREPARATION - TRAINING SPLIT ONLY")
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

df = pd.read_csv(TRAIN_FILE)

required_columns = {
    "filename",
    "original_class",
    "target",
    "leakage_group",
}

missing = sorted(required_columns - set(df.columns))

if missing:
    raise ValueError(
        f"train.csv missing required columns: {missing}"
    )

if df["leakage_group"].isna().any():
    raise ValueError("Missing leakage_group values detected.")

if not set(pd.unique(df["target"])).issubset({0, 1}):
    raise ValueError("Target must contain only 0/1.")

image_column = resolve_image_column(df)

print(f"Training rows:       {len(df)}")
print(f"Image path column:   {image_column}")
print(f"Unique groups:       {df['leakage_group'].nunique()}")
print()
print("Original class distribution:")
print(df["original_class"].value_counts())
print()
print("Binary target distribution:")
print(df["target"].value_counts().sort_index())


# ============================================================
# IMAGE FILE AUDIT
# ============================================================

print()
print("=" * 92)
print("IMAGE FILE AUDIT")
print("=" * 92)

audit_rows = []
shape_counts = {}

total = len(df)

for i, (_, row) in enumerate(df.iterrows(), start=1):
    ok, status, shape = verify_image(row[image_column])

    if shape is not None:
        shape_counts[str(shape)] = shape_counts.get(str(shape), 0) + 1

    if not ok:
        audit_rows.append({
            "filename": row["filename"],
            "original_class": row["original_class"],
            "target": int(row["target"]),
            "leakage_group": row["leakage_group"],
            "image_path": str(row[image_column]),
            "status": status,
        })

    if i % 2000 == 0 or i == total:
        print(f"Checked {i}/{total}")

audit_df = pd.DataFrame(audit_rows)

if len(audit_df) > 0:
    audit_df.to_csv(MISSING_FILE, index=False)

    print()
    print(f"Invalid/missing images: {len(audit_df)}")
    print(f"Saved: {MISSING_FILE}")

    raise RuntimeError(
        "V3 dataset preparation stopped because some training "
        "images are missing or unreadable."
    )

print()
print("All training images readable: PASS")
print("Observed image shapes:")
for shape, count in sorted(
    shape_counts.items(),
    key=lambda x: x[1],
    reverse=True,
):
    print(f"  {shape}: {count}")


# ============================================================
# LOCK GROUP-AWARE CV FOLDS
# ============================================================

print()
print("=" * 92)
print("CREATING LOCKED 3-FOLD GROUP-AWARE CV")
print("=" * 92)

X_dummy = np.zeros((len(df), 1), dtype=np.uint8)
y = df["target"].astype(int).to_numpy()
groups = df["leakage_group"].astype(str).to_numpy()

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

fold_assignment = np.full(
    len(df),
    fill_value=-1,
    dtype=int,
)

fold_summaries = []

for fold_index, (fit_idx, val_idx) in enumerate(
    cv.split(
        X_dummy,
        y,
        groups=groups,
    ),
    start=1,
):
    fit_groups = set(groups[fit_idx])
    val_groups = set(groups[val_idx])

    overlap = fit_groups & val_groups

    if overlap:
        raise RuntimeError(
            f"Fold {fold_index}: leakage-group overlap detected."
        )

    fold_assignment[val_idx] = fold_index

    fold_df = df.iloc[val_idx]

    summary = {
        "fold": int(fold_index),
        "rows": int(len(val_idx)),
        "unique_groups": int(
            fold_df["leakage_group"].nunique()
        ),
        "normal": int(
            (fold_df["target"] == 0).sum()
        ),
        "early_demented": int(
            (fold_df["target"] == 1).sum()
        ),
        "NonDemented": int(
            (fold_df["original_class"] == "NonDemented").sum()
        ),
        "VeryMildDemented": int(
            (
                fold_df["original_class"]
                == "VeryMildDemented"
            ).sum()
        ),
        "MildDemented": int(
            (
                fold_df["original_class"]
                == "MildDemented"
            ).sum()
        ),
        "group_overlap_with_fit": int(len(overlap)),
    }

    fold_summaries.append(summary)

    print()
    print(f"Fold {fold_index}")
    print(f"  rows:            {summary['rows']}")
    print(f"  unique groups:   {summary['unique_groups']}")
    print(f"  Normal:          {summary['normal']}")
    print(f"  Early_Demented:  {summary['early_demented']}")
    print(f"  NonDemented:     {summary['NonDemented']}")
    print(
        f"  VeryMild:        "
        f"{summary['VeryMildDemented']}"
    )
    print(
        f"  Mild:            "
        f"{summary['MildDemented']}"
    )
    print("  leakage overlap: 0")


if np.any(fold_assignment < 1):
    raise RuntimeError(
        "Some rows did not receive a CV fold assignment."
    )


# ============================================================
# SAVE MANIFEST
# ============================================================

manifest = df.copy()

manifest["v3_cv_fold"] = fold_assignment
manifest["v3_image_path"] = (
    manifest[image_column].astype(str)
)

# Keep the most important fields first.
preferred_first = [
    "filename",
    "original_class",
    "target",
    "target_name",
    "leakage_group",
    "v3_cv_fold",
    "v3_image_path",
]

existing_first = [
    c for c in preferred_first
    if c in manifest.columns
]

remaining = [
    c for c in manifest.columns
    if c not in existing_first
]

manifest = manifest[
    existing_first + remaining
]

manifest.to_csv(
    MANIFEST_FILE,
    index=False,
)


# ============================================================
# FINAL QC
# ============================================================

# Every leakage group must belong to exactly one CV fold.
group_fold_counts = (
    manifest
    .groupby("leakage_group")["v3_cv_fold"]
    .nunique()
)

groups_crossing_folds = int(
    (group_fold_counts > 1).sum()
)

if groups_crossing_folds != 0:
    raise RuntimeError(
        f"{groups_crossing_folds} leakage groups cross CV folds."
    )

summary = {
    "status": "PASS",
    "purpose": (
        "Locked training-only group-aware CV manifest "
        "for V3 CNN development."
    ),
    "data_policy": {
        "train_loaded": True,
        "validation_loaded": False,
        "test_loaded": False,
    },
    "source_train_file": str(TRAIN_FILE),
    "source_train_sha256": sha256_file(TRAIN_FILE),
    "image_column_used": image_column,
    "rows": int(len(manifest)),
    "unique_leakage_groups": int(
        manifest["leakage_group"].nunique()
    ),
    "cv": {
        "type": "StratifiedGroupKFold",
        "n_splits": N_SPLITS,
        "shuffle": True,
        "random_state": RANDOM_STATE,
    },
    "groups_crossing_folds": groups_crossing_folds,
    "image_shapes": shape_counts,
    "class_distribution": {
        str(k): int(v)
        for k, v in (
            manifest["original_class"]
            .value_counts()
            .to_dict()
            .items()
        )
    },
    "binary_distribution": {
        str(k): int(v)
        for k, v in (
            manifest["target"]
            .value_counts()
            .sort_index()
            .to_dict()
            .items()
        )
    },
    "folds": fold_summaries,
    "manifest_file": str(MANIFEST_FILE),
    "manifest_sha256": sha256_file(MANIFEST_FILE),
    "important": (
        "This manifest is for V3 development only. "
        "Do not use the existing project test set for V3 tuning."
    ),
}

with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=4,
    )

text_lines = [
    "=" * 92,
    "V3 CNN DATASET PREPARATION",
    "=" * 92,
    "",
    "STATUS: PASS",
    "",
    "DATA POLICY",
    "Train:      used",
    "Validation: NOT loaded",
    "Test:       NOT loaded",
    "",
    f"Rows: {len(manifest)}",
    (
        "Unique leakage groups: "
        f"{manifest['leakage_group'].nunique()}"
    ),
    f"Image path column: {image_column}",
    f"Groups crossing CV folds: {groups_crossing_folds}",
    "",
    "LOCKED CV",
    "StratifiedGroupKFold",
    f"n_splits = {N_SPLITS}",
    f"random_state = {RANDOM_STATE}",
    "",
]

for fold in fold_summaries:
    text_lines.extend([
        f"Fold {fold['fold']}:",
        f"  rows = {fold['rows']}",
        f"  Normal = {fold['normal']}",
        f"  Early_Demented = {fold['early_demented']}",
        (
            "  VeryMildDemented = "
            f"{fold['VeryMildDemented']}"
        ),
        (
            "  MildDemented = "
            f"{fold['MildDemented']}"
        ),
        "",
    ])

SUMMARY_TXT.write_text(
    "\n".join(text_lines),
    encoding="utf-8",
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 92)
print("V3 DATASET PREPARATION COMPLETE")
print("=" * 92)
print()
print("STATUS: PASS")
print(f"Rows: {len(manifest)}")
print(
    "Groups crossing CV folds: "
    f"{groups_crossing_folds}"
)
print()
print("Existing validation/test were NOT loaded.")
print()
print("Saved:")
print(MANIFEST_FILE)
print(SUMMARY_JSON)
print(SUMMARY_TXT)
