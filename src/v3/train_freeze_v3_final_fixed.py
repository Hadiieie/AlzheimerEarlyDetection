from pathlib import Path
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ============================================================
# TRAIN + FREEZE FINAL V3
# Alzheimer MRI Early Detection
#
# PURPOSE
# -------
# Train ONE deployable V3 ResNet18 on ALL locked V3 development
# rows using the already-selected training recipe, then freeze
# the model BEFORE any untouched evaluation.
#
# IMPORTANT
# ---------
# - Uses ONLY:
#       results/V3_CNN_DATASET/v3_training_cv_manifest.csv
# - Reads the EXACT threshold from:
#       results/V3_OOF_THRESHOLD/v3_selected_threshold.json
# - DOES NOT hard-code/round the threshold.
# - DOES NOT load old validation.csv.
# - DOES NOT load old test.csv.
# - DOES NOT load any new untouched evaluation data.
# - DOES NOT tune architecture, epochs, LR, weighting, or threshold.
#
# Locked training recipe:
#   Stage 1: classifier head only, 3 epochs, LR=1e-3
#   Stage 2: layer4 + classifier head, 3 epochs, LR=1e-4
#
# The threshold is taken EXACTLY from the OOF threshold artifact.
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

V3_DATA_DIR = PROJECT_ROOT / "results" / "V3_CNN_DATASET"
V3_BASELINE_DIR = PROJECT_ROOT / "results" / "V3_RESNET18_BASELINE"
V3_THRESHOLD_DIR = PROJECT_ROOT / "results" / "V3_OOF_THRESHOLD"
FINAL_DIR = PROJECT_ROOT / "results" / "FINAL_MODEL_V3"

FINAL_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_FILE = V3_DATA_DIR / "v3_training_cv_manifest.csv"

THRESHOLD_FILE = (
    V3_THRESHOLD_DIR
    / "v3_selected_threshold.json"
)

BASELINE_SUMMARY_FILE = (
    V3_BASELINE_DIR
    / "v3_resnet18_summary.json"
)

FINAL_MODEL_FILE = (
    FINAL_DIR
    / "final_v3_resnet18.pt"
)

FINAL_CONFIG_FILE = (
    FINAL_DIR
    / "final_v3_config.json"
)

FINAL_TRAIN_LOG_FILE = (
    FINAL_DIR
    / "final_v3_training_log.csv"
)

FREEZE_MANIFEST_FILE = (
    FINAL_DIR
    / "freeze_manifest_v3.json"
)

CHECKSUM_FILE = (
    FINAL_DIR
    / "SHA256_CHECKSUMS_V3.txt"
)


# ============================================================
# 2. LOCKED V3 TRAINING SPECIFICATION
# ============================================================

RANDOM_SEED = 42

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0   # safest on Windows

HEAD_EPOCHS = 3
LAYER4_EPOCHS = 3

HEAD_LR = 1e-3
LAYER4_LR = 1e-4

WEIGHT_DECAY = 1e-4

EXPECTED_DEVELOPMENT_ROWS = 20891

DEVICE = torch.device("cpu")


# ============================================================
# 3. TARGET METRICS THAT WERE REQUIRED > 90% IN OOF
# ============================================================

REQUIRED_OOF_METRICS = [
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


# ============================================================
# 4. UTILITY FUNCTIONS
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CPU-only project.
    # Use most logical CPU cores while leaving one core free.
    threads = max(
        1,
        (os.cpu_count() or 2) - 1,
    )

    torch.set_num_threads(threads)


def validate_image_files(df: pd.DataFrame) -> None:
    """
    Fast preflight check that every V3 image path still exists.
    We do not decode all 20,891 images here because that was already
    done during V3 dataset preparation.
    """
    missing = []

    for path_string in df["v3_image_path"].astype(str):
        path = Path(path_string)

        if not path.exists():
            missing.append(path_string)

            if len(missing) >= 20:
                break

    if missing:
        print()
        print("Missing image examples:")

        for item in missing:
            print(item)

        raise FileNotFoundError(
            "One or more V3 training image files are missing."
        )


# ============================================================
# 5. PYTORCH DATASET
# ============================================================

class MRIDataset(Dataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform,
    ):
        self.df = (
            dataframe
            .reset_index(drop=True)
            .copy()
        )

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_path = Path(
            str(row["v3_image_path"])
        )

        image = Image.open(
            image_path
        ).convert("L")

        # ImageNet-pretrained ResNet18 expects 3 channels.
        image = image.convert("RGB")

        image = self.transform(
            image
        )

        target = int(
            row["target"]
        )

        return image, target


# ============================================================
# 6. LOCKED INPUT TRANSFORM
# ============================================================

# Same training augmentation used during V3 CV.
#
# No horizontal flip:
# left/right anatomical information may matter.

train_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        antialias=True,
    ),
    transforms.RandomRotation(
        degrees=3,
        fill=0,
    ),
    transforms.ColorJitter(
        brightness=0.05,
        contrast=0.05,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
    ),
])


# ============================================================
# 7. MODEL / TRAINING HELPERS
# ============================================================

def compute_binary_class_weights(
    y: np.ndarray,
):
    y = np.asarray(
        y,
        dtype=int,
    )

    counts = np.bincount(
        y,
        minlength=2,
    )

    if np.any(counts == 0):
        raise ValueError(
            f"Both binary classes must be present. Counts={counts.tolist()}"
        )

    total = counts.sum()

    weights = (
        total
        / (
            2.0
            * counts.astype(float)
        )
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE,
    )


def build_model():
    """
    Build the same ImageNet-pretrained ResNet18 architecture
    used during V3 group-aware cross-validation.
    """
    weights = (
        models
        .ResNet18_Weights
        .DEFAULT
    )

    model = models.resnet18(
        weights=weights
    )

    # Initially freeze whole backbone.
    for param in model.parameters():
        param.requires_grad = False

    in_features = (
        model.fc.in_features
    )

    model.fc = nn.Linear(
        in_features,
        2,
    )

    return model.to(
        DEVICE
    )


def set_head_only(
    model,
):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True


def set_layer4_and_head(
    model,
):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True


def make_optimizer(
    model,
    learning_rate,
):
    trainable_parameters = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    if not trainable_parameters:
        raise RuntimeError(
            "No trainable parameters were found."
        )

    return torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
):
    model.train()

    total_loss = 0.0
    correct = 0
    seen = 0

    for images, targets in loader:
        images = images.to(
            DEVICE
        )

        targets = targets.to(
            DEVICE
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            images
        )

        loss = criterion(
            logits,
            targets,
        )

        loss.backward()

        optimizer.step()

        batch_size = int(
            targets.size(0)
        )

        total_loss += (
            float(loss.item())
            * batch_size
        )

        predictions = (
            logits.argmax(
                dim=1
            )
        )

        correct += int(
            (
                predictions
                == targets
            )
            .sum()
            .item()
        )

        seen += batch_size

    if seen == 0:
        raise RuntimeError(
            "Training DataLoader produced zero samples."
        )

    return {
        "loss": float(
            total_loss
            / seen
        ),
        "training_accuracy": float(
            correct
            / seen
        ),
    }


# ============================================================
# 8. HEADER
# ============================================================

print()
print("=" * 100)
print("TRAIN + FREEZE FINAL V3")
print("=" * 100)
print()

print("PyTorch version:")
print(torch.__version__)

print()
print("Device:")
print(DEVICE)

print()
print("DATA POLICY")
print("-----------")
print("Locked V3 development manifest: LOADED")
print("Old validation.csv:             NOT LOADED")
print("Old test.csv:                   NOT LOADED")
print("New untouched evaluation data:  NOT LOADED")
print()


# ============================================================
# 9. VERIFY REQUIRED ARTIFACTS
# ============================================================

required_files = [
    MANIFEST_FILE,
    THRESHOLD_FILE,
    BASELINE_SUMMARY_FILE,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found:\n{path}"
        )

print("Required development artifacts: PASS")


# ============================================================
# 10. LOAD EXACT OOF-SELECTED THRESHOLD
# ============================================================

with open(
    THRESHOLD_FILE,
    "r",
    encoding="utf-8",
) as f:
    threshold_info = json.load(
        f
    )

threshold_status = str(
    threshold_info.get(
        "status",
        "",
    )
)

if threshold_status != "GOAL_ACHIEVED_WITH_THRESHOLD":
    raise RuntimeError(
        "Threshold artifact status is not "
        "'GOAL_ACHIEVED_WITH_THRESHOLD'. "
        f"Found: {threshold_status}"
    )

selected_threshold = float(
    threshold_info[
        "selected_threshold"
    ]
)

if not (
    0.0
    < selected_threshold
    < 1.0
):
    raise ValueError(
        "Selected threshold must be within (0,1). "
        f"Found: {selected_threshold}"
    )

# IMPORTANT FIX:
# We DO NOT compare against a rounded manually typed value.
# The exact JSON artifact value becomes the official locked threshold.
LOCKED_THRESHOLD = selected_threshold

selected_oof_metrics = (
    threshold_info.get(
        "selected_metrics",
        {},
    )
)

missing_metrics = [
    metric
    for metric in REQUIRED_OOF_METRICS
    if metric not in selected_oof_metrics
]

if missing_metrics:
    raise RuntimeError(
        "Threshold artifact is missing required OOF metrics: "
        f"{missing_metrics}"
    )

failed_oof_metrics = {
    metric: float(
        selected_oof_metrics[
            metric
        ]
    )
    for metric in REQUIRED_OOF_METRICS
    if float(
        selected_oof_metrics[
            metric
        ]
    ) <= 0.90
}

if failed_oof_metrics:
    raise RuntimeError(
        "Threshold artifact no longer satisfies the locked "
        f">90% OOF target: {failed_oof_metrics}"
    )

print()
print("OOF threshold artifact verification: PASS")

print(
    "Exact locked threshold:"
)

print(
    f"{LOCKED_THRESHOLD:.17f}"
)

print()
print(
    "All requested OOF development metrics "
    "are strictly above 90%: PASS"
)


# ============================================================
# 11. VERIFY V3 BASELINE DATA POLICY
# ============================================================

with open(
    BASELINE_SUMMARY_FILE,
    "r",
    encoding="utf-8",
) as f:
    baseline_summary = json.load(
        f
    )

data_policy = baseline_summary.get(
    "data_policy",
    {},
)

if data_policy.get(
    "old_validation_loaded",
    True,
):
    raise RuntimeError(
        "Baseline summary does not confirm that "
        "old validation.csv remained untouched."
    )

if data_policy.get(
    "old_test_loaded",
    True,
):
    raise RuntimeError(
        "Baseline summary does not confirm that "
        "old test.csv remained untouched."
    )

print(
    "V3 development data-policy verification: PASS"
)


# ============================================================
# 12. LOAD LOCKED DEVELOPMENT MANIFEST
# ============================================================

df = pd.read_csv(
    MANIFEST_FILE
)

required_columns = {
    "filename",
    "original_class",
    "target",
    "leakage_group",
    "v3_cv_fold",
    "v3_image_path",
}

missing_columns = sorted(
    required_columns
    - set(df.columns)
)

if missing_columns:
    raise ValueError(
        "V3 manifest missing required columns: "
        f"{missing_columns}"
    )

if len(df) != EXPECTED_DEVELOPMENT_ROWS:
    raise ValueError(
        f"Expected {EXPECTED_DEVELOPMENT_ROWS} V3 rows, "
        f"found {len(df)}."
    )

if df["v3_image_path"].isna().any():
    raise ValueError(
        "Missing v3_image_path values found."
    )

if df["leakage_group"].isna().any():
    raise ValueError(
        "Missing leakage_group values found."
    )

targets = (
    df["target"]
    .astype(int)
    .to_numpy()
)

if not set(
    np.unique(
        targets
    )
).issubset(
    {0, 1}
):
    raise ValueError(
        "Target must contain only 0/1."
    )

validate_image_files(
    df
)

print()
print("Full V3 development manifest verification: PASS")

print(
    f"Rows: "
    f"{len(df)}"
)

print(
    f"Unique leakage groups: "
    f"{df['leakage_group'].nunique()}"
)

print(
    f"Normal: "
    f"{int((targets == 0).sum())}"
)

print(
    f"Early_Demented: "
    f"{int((targets == 1).sum())}"
)


# ============================================================
# 13. BUILD DATA LOADER
# ============================================================

set_seed(
    RANDOM_SEED
)

dataset = MRIDataset(
    df,
    train_transform,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=False,
    drop_last=False,
)

class_weights = (
    compute_binary_class_weights(
        targets
    )
)

print()
print(
    "Binary class weights:"
)

print(
    class_weights
    .cpu()
    .numpy()
)


# ============================================================
# 14. BUILD MODEL
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

model = build_model()

training_log = []


# ============================================================
# 15. STAGE 1 - HEAD ONLY
# ============================================================

print()
print("=" * 100)
print("STAGE 1: CLASSIFIER HEAD - FIXED 3 EPOCHS")
print("=" * 100)

set_head_only(
    model
)

optimizer = make_optimizer(
    model,
    HEAD_LR,
)

for epoch in range(
    1,
    HEAD_EPOCHS + 1,
):
    start_time = (
        time.perf_counter()
    )

    metrics = train_one_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        optimizer=optimizer,
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    row = {
        "stage": "HEAD",
        "epoch": int(epoch),
        "loss": float(
            metrics["loss"]
        ),
        "training_accuracy": float(
            metrics[
                "training_accuracy"
            ]
        ),
        "seconds": float(
            elapsed
        ),
    }

    training_log.append(
        row
    )

    print(
        f"HEAD epoch "
        f"{epoch}/{HEAD_EPOCHS} | "
        f"loss={row['loss']:.5f} | "
        f"train_acc="
        f"{row['training_accuracy']:.5f} | "
        f"{elapsed:.1f}s"
    )


# ============================================================
# 16. STAGE 2 - LAYER4 + HEAD
# ============================================================

print()
print("=" * 100)
print("STAGE 2: LAYER4 + HEAD - FIXED 3 EPOCHS")
print("=" * 100)

set_layer4_and_head(
    model
)

optimizer = make_optimizer(
    model,
    LAYER4_LR,
)

for epoch in range(
    1,
    LAYER4_EPOCHS + 1,
):
    start_time = (
        time.perf_counter()
    )

    metrics = train_one_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        optimizer=optimizer,
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    row = {
        "stage": "LAYER4",
        "epoch": int(epoch),
        "loss": float(
            metrics["loss"]
        ),
        "training_accuracy": float(
            metrics[
                "training_accuracy"
            ]
        ),
        "seconds": float(
            elapsed
        ),
    }

    training_log.append(
        row
    )

    print(
        f"LAYER4 epoch "
        f"{epoch}/{LAYER4_EPOCHS} | "
        f"loss={row['loss']:.5f} | "
        f"train_acc="
        f"{row['training_accuracy']:.5f} | "
        f"{elapsed:.1f}s"
    )


# ============================================================
# 17. SAVE FINAL DEPLOYABLE MODEL
# ============================================================

training_log_df = pd.DataFrame(
    training_log
)

training_log_df.to_csv(
    FINAL_TRAIN_LOG_FILE,
    index=False,
)

checkpoint = {
    "version": "V3",
    "status": (
        "FROZEN_BEFORE_UNTOUCHED_EVALUATION"
    ),
    "architecture": "resnet18",
    "weights_origin": (
        "ImageNet pretrained"
    ),
    "num_classes": 2,
    "class_index": {
        "0": "Normal",
        "1": "Early_Demented",
    },
    "state_dict": (
        model.state_dict()
    ),
    "selected_threshold": float(
        LOCKED_THRESHOLD
    ),
    "input": {
        "image_size": IMAGE_SIZE,
        "source_channels": 1,
        "model_channels": 3,
        "grayscale_to_rgb": True,
        "normalization_mean": [
            0.485,
            0.456,
            0.406,
        ],
        "normalization_std": [
            0.229,
            0.224,
            0.225,
        ],
    },
    "training": {
        "development_rows": int(
            len(df)
        ),
        "unique_leakage_groups": int(
            df[
                "leakage_group"
            ].nunique()
        ),
        "random_seed": RANDOM_SEED,
        "batch_size": BATCH_SIZE,
        "head_epochs": HEAD_EPOCHS,
        "layer4_epochs": (
            LAYER4_EPOCHS
        ),
        "head_lr": HEAD_LR,
        "layer4_lr": LAYER4_LR,
        "weight_decay": (
            WEIGHT_DECAY
        ),
        "class_weights": (
            class_weights
            .cpu()
            .numpy()
            .tolist()
        ),
        "augmentation": {
            "resize": [
                IMAGE_SIZE,
                IMAGE_SIZE,
            ],
            "rotation_degrees": 3,
            "brightness": 0.05,
            "contrast": 0.05,
            "horizontal_flip": False,
        },
    },
}

torch.save(
    checkpoint,
    FINAL_MODEL_FILE,
)


# ============================================================
# 18. SAVE FINAL CONFIG
# ============================================================

threshold_independent = (
    threshold_info.get(
        "threshold_independent_metrics",
        {},
    )
)

config = {
    "version": "V3",
    "status": (
        "FROZEN_BEFORE_UNTOUCHED_EVALUATION"
    ),
    "task": (
        "Normal vs Early_Demented MRI classification"
    ),
    "model": {
        "architecture": "ResNet18",
        "pretraining": "ImageNet",
        "classes": [
            "Normal",
            "Early_Demented",
        ],
    },
    "input": {
        "image_size": 224,
        "source": (
            "processed 2D grayscale MRI"
        ),
        "conversion": (
            "grayscale duplicated to RGB"
        ),
        "normalization_mean": [
            0.485,
            0.456,
            0.406,
        ],
        "normalization_std": [
            0.229,
            0.224,
            0.225,
        ],
    },
    "training_data": {
        "rows": int(
            len(df)
        ),
        "unique_leakage_groups": int(
            df[
                "leakage_group"
            ].nunique()
        ),
        "source_manifest": str(
            MANIFEST_FILE
        ),
    },
    "training_recipe": {
        "stage_1": {
            "trainable": (
                "classifier head only"
            ),
            "epochs": HEAD_EPOCHS,
            "learning_rate": (
                HEAD_LR
            ),
        },
        "stage_2": {
            "trainable": (
                "layer4 + classifier head"
            ),
            "epochs": (
                LAYER4_EPOCHS
            ),
            "learning_rate": (
                LAYER4_LR
            ),
        },
        "optimizer": "AdamW",
        "weight_decay": (
            WEIGHT_DECAY
        ),
        "class_weighting": (
            "balanced binary "
            "CrossEntropyLoss"
        ),
        "batch_size": (
            BATCH_SIZE
        ),
        "augmentation": {
            "rotation_degrees": 3,
            "brightness": 0.05,
            "contrast": 0.05,
            "horizontal_flip": False,
        },
    },
    "selected_threshold": float(
        LOCKED_THRESHOLD
    ),
    "threshold_source": (
        "Exhaustive threshold search "
        "on leakage-safe 3-fold V3 OOF predictions"
    ),
    "development_oof": {
        "selected_metrics": (
            selected_oof_metrics
        ),
        "roc_auc": (
            threshold_independent.get(
                "roc_auc"
            )
        ),
        "pr_auc": (
            threshold_independent.get(
                "pr_auc"
            )
        ),
    },
    "scientific_scope": (
        "The >90% values are leakage-safe out-of-fold "
        "development metrics. They are not yet untouched "
        "patient-level or external validation performance."
    ),
    "locked_components": [
        "architecture",
        "input preprocessing",
        "training recipe",
        "class weighting",
        "decision threshold",
    ],
}

with open(
    FINAL_CONFIG_FILE,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        config,
        f,
        indent=4,
    )


# ============================================================
# 19. FREEZE MANIFEST + CHECKSUMS
# ============================================================

artifact_paths = [
    FINAL_MODEL_FILE,
    FINAL_CONFIG_FILE,
    FINAL_TRAIN_LOG_FILE,
]

hashes = {
    path.name: sha256_file(
        path
    )
    for path in artifact_paths
}

freeze_manifest = {
    "version": "V3",
    "status": (
        "FROZEN_BEFORE_UNTOUCHED_EVALUATION"
    ),
    "python_version": (
        sys.version
    ),
    "torch_version": (
        torch.__version__
    ),
    "selected_threshold": float(
        LOCKED_THRESHOLD
    ),
    "development_manifest": str(
        MANIFEST_FILE
    ),
    "threshold_artifact": str(
        THRESHOLD_FILE
    ),
    "baseline_summary": str(
        BASELINE_SUMMARY_FILE
    ),
    "artifacts": {
        "model": (
            FINAL_MODEL_FILE.name
        ),
        "config": (
            FINAL_CONFIG_FILE.name
        ),
        "training_log": (
            FINAL_TRAIN_LOG_FILE.name
        ),
    },
    "sha256": hashes,
    "old_validation_loaded": False,
    "old_test_loaded": False,
    "untouched_evaluation_loaded": False,
    "important": (
        "Do not alter the frozen V3 architecture, preprocessing, "
        "training recipe, class weighting, or threshold based on "
        "future evaluation results."
    ),
}

with open(
    FREEZE_MANIFEST_FILE,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        freeze_manifest,
        f,
        indent=4,
    )

hashes[
    FREEZE_MANIFEST_FILE.name
] = sha256_file(
    FREEZE_MANIFEST_FILE
)

with open(
    CHECKSUM_FILE,
    "w",
    encoding="utf-8",
) as f:
    for name, digest in hashes.items():
        f.write(
            f"{digest}  {name}\n"
        )


# ============================================================
# 20. FINAL OUTPUT
# ============================================================

print()
print("=" * 100)
print("FINAL V3 FREEZE COMPLETE")
print("=" * 100)
print()

print(
    "V3 is now locked before untouched evaluation."
)

print()
print(
    "Exact frozen threshold:"
)

print(
    f"{LOCKED_THRESHOLD:.17f}"
)

print()
print(
    "Old validation/test were NOT loaded."
)

print(
    "No untouched evaluation data were loaded."
)

print()
print(
    "Final V3 model SHA256:"
)

print(
    hashes[
        FINAL_MODEL_FILE.name
    ]
)

print()
print("Saved:")
print(FINAL_MODEL_FILE)
print(FINAL_CONFIG_FILE)
print(FINAL_TRAIN_LOG_FILE)
print(FREEZE_MANIFEST_FILE)
print(CHECKSUM_FILE)

print()
print(
    "NEXT STEP:"
)

print(
    "Evaluate this exact frozen V3 model on genuinely "
    "untouched MRI subjects/data."
)
