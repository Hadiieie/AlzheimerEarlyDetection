from pathlib import Path
import copy
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

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


# ============================================================
# V3 RESNET18 BASELINE
# Alzheimer MRI Early Detection
#
# DATA POLICY:
# - Uses ONLY the locked V3 training CV manifest.
# - Does NOT load the old validation.csv.
# - Does NOT load the old test.csv.
# - Uses the already-locked 3 group-aware folds.
#
# CPU-ORIENTED DESIGN:
# - Pretrained ResNet18
# - Stage 1: classifier head only
# - Stage 2: unfreeze layer4 + classifier head
# - Small/controlled augmentation
# - Binary class weighting
# - Best checkpoint per fold selected on fold validation BA
#
# IMPORTANT:
# This is a V3 development benchmark, NOT final performance.
# ============================================================


# ============================================================
# 1. PATHS / CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_FILE = (
    PROJECT_ROOT
    / "results"
    / "V3_CNN_DATASET"
    / "v3_training_cv_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "V3_RESNET18_BASELINE"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OOF_FILE = OUTPUT_DIR / "v3_resnet18_oof_predictions.csv"
FOLD_METRICS_FILE = OUTPUT_DIR / "v3_resnet18_fold_metrics.csv"
SUMMARY_JSON = OUTPUT_DIR / "v3_resnet18_summary.json"
SUMMARY_TXT = OUTPUT_DIR / "v3_resnet18_summary.txt"

RANDOM_SEED = 42

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0  # safest on Windows

HEAD_EPOCHS = 3
LAYER4_EPOCHS = 3

HEAD_LR = 1e-3
LAYER4_LR = 1e-4

WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 2

DEVICE = torch.device("cpu")


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CPU deterministic behavior
    torch.set_num_threads(
        max(1, (os.cpu_count() or 2) - 1)
    )


set_seed(RANDOM_SEED)


# ============================================================
# 3. DATASET
# ============================================================

class MRIDataset(Dataset):
    def __init__(
        self,
        dataframe,
        image_transform,
    ):
        self.df = dataframe.reset_index(drop=True).copy()
        self.transform = image_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_path = Path(
            str(row["v3_image_path"])
        )

        image = Image.open(image_path).convert("L")

        # ResNet18 pretrained weights expect 3 channels.
        image = image.convert("RGB")

        image = self.transform(image)

        target = int(row["target"])

        return (
            image,
            target,
            int(index),
        )


# ============================================================
# 4. TRANSFORMS
# ============================================================

# MRI-safe-ish augmentation:
# - tiny rotation only
# - slight brightness / contrast perturbation
# - NO horizontal flip because left/right anatomy may matter
#
# ImageNet normalization is used because pretrained ResNet18
# weights were trained with that convention.

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
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        antialias=True,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# 5. HELPERS
# ============================================================

def compute_binary_class_weights(y):
    y = np.asarray(y, dtype=int)

    counts = np.bincount(
        y,
        minlength=2,
    )

    total = counts.sum()

    weights = total / (
        2.0 * counts.astype(float)
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE,
    )


def build_model():
    try:
        weights = models.ResNet18_Weights.DEFAULT

        model = models.resnet18(
            weights=weights
        )

    except Exception as exc:
        raise RuntimeError(
            "\nCould not load/download pretrained ResNet18 weights.\n"
            "The first run may need an internet connection so torchvision "
            "can download the pretrained weights.\n"
            f"Original error:\n{exc}"
        )

    # Freeze entire backbone first.
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        2,
    )

    return model.to(DEVICE)


def set_trainable_head_only(model):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True


def set_trainable_layer4_and_head(model):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True


def make_optimizer(model, lr):
    params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    return torch.optim.AdamW(
        params,
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )


def metrics_from_probs(
    y_true,
    probs,
    threshold=0.5,
):
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)

    pred = (
        probs >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn)
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp)
        else np.nan
    )

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                pred,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                pred,
            )
        ),
        "sensitivity": float(
            sensitivity
        ),
        "specificity": float(
            specificity
        ),
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
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probs,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probs,
            )
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def subgroup_metrics(
    dataframe,
    probs,
    threshold=0.5,
):
    pred = (
        np.asarray(probs)
        >= threshold
    ).astype(int)

    classes = (
        dataframe["original_class"]
        .astype(str)
        .str.strip()
        .str.lower()
        .to_numpy()
    )

    verymild = (
        classes == "verymilddemented"
    )

    mild = (
        classes == "milddemented"
    )

    normal = (
        classes == "nondemented"
    )

    return {
        "verymild_sensitivity": (
            float(np.mean(pred[verymild] == 1))
            if verymild.any()
            else None
        ),
        "mild_sensitivity": (
            float(np.mean(pred[mild] == 1))
            if mild.any()
            else None
        ),
        "nondemented_specificity": (
            float(np.mean(pred[normal] == 0))
            if normal.any()
            else None
        ),
        "verymild_n": int(
            verymild.sum()
        ),
        "mild_n": int(
            mild.sum()
        ),
        "nondemented_n": int(
            normal.sum()
        ),
    }


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
):
    model.train()

    total_loss = 0.0
    seen = 0

    for images, targets, _ in loader:
        images = images.to(
            DEVICE,
            non_blocking=False,
        )

        targets = targets.to(
            DEVICE,
            non_blocking=False,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(images)

        loss = criterion(
            logits,
            targets,
        )

        loss.backward()

        optimizer.step()

        batch_size = targets.size(0)

        total_loss += (
            float(loss.item())
            * batch_size
        )

        seen += batch_size

    return (
        total_loss / seen
        if seen
        else math.nan
    )


@torch.no_grad()
def predict_loader(
    model,
    loader,
):
    model.eval()

    all_targets = []
    all_probs = []
    all_local_indices = []

    for images, targets, local_indices in loader:
        images = images.to(
            DEVICE,
            non_blocking=False,
        )

        logits = model(images)

        probs = torch.softmax(
            logits,
            dim=1,
        )[:, 1]

        all_targets.extend(
            targets.numpy().tolist()
        )

        all_probs.extend(
            probs.cpu().numpy().tolist()
        )

        all_local_indices.extend(
            local_indices.numpy().tolist()
        )

    return (
        np.asarray(all_targets, dtype=int),
        np.asarray(all_probs, dtype=float),
        np.asarray(all_local_indices, dtype=int),
    )


def run_stage(
    model,
    train_loader,
    val_loader,
    criterion,
    epochs,
    lr,
    stage_name,
):
    optimizer = make_optimizer(
        model,
        lr,
    )

    best_state = copy.deepcopy(
        model.state_dict()
    )

    best_ba = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0

    history = []

    for epoch in range(
        1,
        epochs + 1,
    ):
        start = time.perf_counter()

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        y_val, p_val, _ = predict_loader(
            model,
            val_loader,
        )

        val_metrics = metrics_from_probs(
            y_val,
            p_val,
            threshold=0.5,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        history.append({
            "stage": stage_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_balanced_accuracy": (
                val_metrics[
                    "balanced_accuracy"
                ]
            ),
            "val_accuracy": (
                val_metrics[
                    "accuracy"
                ]
            ),
            "val_roc_auc": (
                val_metrics[
                    "roc_auc"
                ]
            ),
            "seconds": elapsed,
        })

        print(
            f"    {stage_name} "
            f"epoch {epoch}/{epochs} | "
            f"loss={train_loss:.5f} | "
            f"BA={val_metrics['balanced_accuracy']:.5f} | "
            f"AUC={val_metrics['roc_auc']:.5f} | "
            f"{elapsed:.1f}s"
        )

        current_ba = (
            val_metrics[
                "balanced_accuracy"
            ]
        )

        if (
            current_ba
            > best_ba + 1e-6
        ):
            best_ba = current_ba
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= EARLY_STOP_PATIENCE
        ):
            print(
                f"    Early stopping "
                f"{stage_name}."
            )
            break

    model.load_state_dict(
        best_state
    )

    return {
        "best_balanced_accuracy": float(
            best_ba
        ),
        "best_epoch": int(
            best_epoch
        ),
        "history": history,
    }


# ============================================================
# 6. LOAD LOCKED V3 MANIFEST
# ============================================================

print()
print("=" * 96)
print("V3 RESNET18 BASELINE - CPU")
print("=" * 96)
print()
print("PyTorch:", torch.__version__)
print("Device:", DEVICE)
print()
print("DATA POLICY")
print("train-only locked V3 manifest: LOADED")
print("old validation.csv: NOT LOADED")
print("old test.csv:       NOT LOADED")
print()

if not MANIFEST_FILE.exists():
    raise FileNotFoundError(
        MANIFEST_FILE
    )

manifest = pd.read_csv(
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

missing = sorted(
    required_columns
    - set(manifest.columns)
)

if missing:
    raise ValueError(
        f"Manifest missing columns: {missing}"
    )

fold_values = sorted(
    manifest[
        "v3_cv_fold"
    ].unique().tolist()
)

if fold_values != [1, 2, 3]:
    raise ValueError(
        "Expected locked V3 folds exactly [1, 2, 3]."
    )

print(
    f"Manifest rows: "
    f"{len(manifest)}"
)
print(
    f"Unique groups: "
    f"{manifest['leakage_group'].nunique()}"
)
print(
    f"Locked folds: "
    f"{fold_values}"
)


# ============================================================
# 7. CROSS-VALIDATION TRAINING
# ============================================================

oof_parts = []
fold_metric_rows = []
full_history = []

for fold in fold_values:
    print()
    print("=" * 96)
    print(f"FOLD {fold}/3")
    print("=" * 96)

    set_seed(
        RANDOM_SEED + int(fold)
    )

    train_df = manifest[
        manifest["v3_cv_fold"]
        != fold
    ].reset_index(drop=True)

    val_df = manifest[
        manifest["v3_cv_fold"]
        == fold
    ].reset_index(drop=True)

    train_groups = set(
        train_df[
            "leakage_group"
        ].astype(str)
    )

    val_groups = set(
        val_df[
            "leakage_group"
        ].astype(str)
    )

    overlap = (
        train_groups
        & val_groups
    )

    if overlap:
        raise RuntimeError(
            f"Fold {fold}: group leakage detected."
        )

    print(
        f"Train rows: "
        f"{len(train_df)}"
    )
    print(
        f"Validation rows: "
        f"{len(val_df)}"
    )
    print(
        "Group overlap: 0"
    )

    train_dataset = MRIDataset(
        train_df,
        train_transform,
    )

    val_dataset = MRIDataset(
        val_df,
        eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
    )

    class_weights = (
        compute_binary_class_weights(
            train_df[
                "target"
            ].astype(int).to_numpy()
        )
    )

    print(
        "Binary class weights:",
        class_weights.cpu().numpy(),
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    model = build_model()

    # --------------------------
    # Stage 1: head only
    # --------------------------

    set_trainable_head_only(
        model
    )

    print()
    print("Stage 1: classifier head")

    stage1 = run_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        epochs=HEAD_EPOCHS,
        lr=HEAD_LR,
        stage_name="HEAD",
    )

    # --------------------------
    # Stage 2: layer4 + head
    # --------------------------

    set_trainable_layer4_and_head(
        model
    )

    print()
    print("Stage 2: layer4 + head")

    stage2 = run_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        epochs=LAYER4_EPOCHS,
        lr=LAYER4_LR,
        stage_name="LAYER4",
    )

    # The stage2 runner starts from the best stage1 model.
    # It keeps the best state seen during stage2.
    # Evaluate that final fold model once.

    y_val, p_val, local_idx = (
        predict_loader(
            model,
            val_loader,
        )
    )

    metrics = metrics_from_probs(
        y_val,
        p_val,
        threshold=0.5,
    )

    # predict_loader returns local rows in the val_df order.
    aligned_val_df = val_df.iloc[
        local_idx
    ].reset_index(drop=True)

    subgroups = subgroup_metrics(
        aligned_val_df,
        p_val,
        threshold=0.5,
    )

    fold_metrics = {
        "fold": int(fold),
        **metrics,
        **subgroups,
        "stage1_best_epoch": (
            stage1[
                "best_epoch"
            ]
        ),
        "stage1_best_ba": (
            stage1[
                "best_balanced_accuracy"
            ]
        ),
        "stage2_best_epoch": (
            stage2[
                "best_epoch"
            ]
        ),
        "stage2_best_ba": (
            stage2[
                "best_balanced_accuracy"
            ]
        ),
    }

    fold_metric_rows.append(
        fold_metrics
    )

    for item in stage1["history"]:
        full_history.append({
            "fold": int(fold),
            **item,
        })

    for item in stage2["history"]:
        full_history.append({
            "fold": int(fold),
            **item,
        })

    oof_fold = aligned_val_df[
        [
            "filename",
            "original_class",
            "target",
            "leakage_group",
            "v3_cv_fold",
            "v3_image_path",
        ]
    ].copy()

    oof_fold[
        "resnet18_probability"
    ] = p_val

    oof_fold[
        "resnet18_prediction"
    ] = (
        p_val >= 0.5
    ).astype(int)

    oof_fold[
        "correct"
    ] = (
        oof_fold[
            "resnet18_prediction"
        ].to_numpy()
        ==
        y_val
    ).astype(int)

    oof_parts.append(
        oof_fold
    )

    checkpoint_file = (
        OUTPUT_DIR
        / f"resnet18_fold{fold}_best.pt"
    )

    torch.save(
        {
            "fold": int(fold),
            "model_state_dict": (
                model.state_dict()
            ),
            "class_weights": (
                class_weights
                .cpu()
                .numpy()
                .tolist()
            ),
            "metrics": fold_metrics,
        },
        checkpoint_file,
    )

    print()
    print(
        f"Fold {fold} result:"
    )
    print(
        f"  BA:          "
        f"{metrics['balanced_accuracy']:.6f}"
    )
    print(
        f"  Accuracy:    "
        f"{metrics['accuracy']:.6f}"
    )
    print(
        f"  Sensitivity: "
        f"{metrics['sensitivity']:.6f}"
    )
    print(
        f"  Specificity: "
        f"{metrics['specificity']:.6f}"
    )
    print(
        f"  ROC-AUC:     "
        f"{metrics['roc_auc']:.6f}"
    )
    print(
        f"  VeryMild:    "
        f"{subgroups['verymild_sensitivity']}"
    )


# ============================================================
# 8. OOF RESULTS
# ============================================================

oof = pd.concat(
    oof_parts,
    axis=0,
    ignore_index=True,
)

if len(oof) != len(manifest):
    raise RuntimeError(
        "OOF row count does not match manifest."
    )

if oof["filename"].duplicated().any():
    raise RuntimeError(
        "Duplicate filenames detected in OOF predictions."
    )

oof = oof.sort_values(
    by=[
        "v3_cv_fold",
        "filename",
    ],
).reset_index(drop=True)

oof.to_csv(
    OOF_FILE,
    index=False,
)

y_oof = (
    oof["target"]
    .astype(int)
    .to_numpy()
)

p_oof = (
    oof[
        "resnet18_probability"
    ]
    .astype(float)
    .to_numpy()
)

overall_metrics = metrics_from_probs(
    y_oof,
    p_oof,
    threshold=0.5,
)

overall_subgroups = subgroup_metrics(
    oof,
    p_oof,
    threshold=0.5,
)

fold_metrics_df = pd.DataFrame(
    fold_metric_rows
)

fold_metrics_df.to_csv(
    FOLD_METRICS_FILE,
    index=False,
)

history_df = pd.DataFrame(
    full_history
)

history_file = (
    OUTPUT_DIR
    / "training_history.csv"
)

history_df.to_csv(
    history_file,
    index=False,
)


# ============================================================
# 9. SUMMARY
# ============================================================

summary = {
    "status": "V3_DEVELOPMENT_ONLY",
    "model": (
        "ImageNet-pretrained ResNet18"
    ),
    "device": str(DEVICE),
    "data_policy": {
        "locked_v3_training_manifest": True,
        "old_validation_loaded": False,
        "old_test_loaded": False,
    },
    "training": {
        "folds": 3,
        "batch_size": BATCH_SIZE,
        "image_size": IMAGE_SIZE,
        "head_epochs_max": HEAD_EPOCHS,
        "layer4_epochs_max": LAYER4_EPOCHS,
        "head_lr": HEAD_LR,
        "layer4_lr": LAYER4_LR,
        "weight_decay": WEIGHT_DECAY,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "augmentation": {
            "rotation_degrees": 3,
            "brightness": 0.05,
            "contrast": 0.05,
            "horizontal_flip": False,
        },
    },
    "oof_metrics_threshold_0.5": {
        **overall_metrics,
        **overall_subgroups,
    },
    "fold_metrics": fold_metric_rows,
    "important": (
        "These are out-of-fold V3 development results on the "
        "locked V3 training manifest. They are not final test "
        "performance and must not be compared as if they were "
        "an untouched external test."
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

lines = [
    "=" * 96,
    "V3 RESNET18 BASELINE SUMMARY",
    "=" * 96,
    "",
    "STATUS: DEVELOPMENT ONLY",
    "",
    "OOF METRICS @ THRESHOLD 0.5",
    (
        "Accuracy: "
        f"{overall_metrics['accuracy']:.6f}"
    ),
    (
        "Balanced Accuracy: "
        f"{overall_metrics['balanced_accuracy']:.6f}"
    ),
    (
        "Sensitivity: "
        f"{overall_metrics['sensitivity']:.6f}"
    ),
    (
        "Specificity: "
        f"{overall_metrics['specificity']:.6f}"
    ),
    (
        "Precision: "
        f"{overall_metrics['precision']:.6f}"
    ),
    (
        "F1: "
        f"{overall_metrics['f1']:.6f}"
    ),
    (
        "ROC-AUC: "
        f"{overall_metrics['roc_auc']:.6f}"
    ),
    (
        "PR-AUC: "
        f"{overall_metrics['pr_auc']:.6f}"
    ),
    (
        "VeryMild sensitivity: "
        f"{overall_subgroups['verymild_sensitivity']}"
    ),
    (
        "Mild sensitivity: "
        f"{overall_subgroups['mild_sensitivity']}"
    ),
    (
        "NonDemented specificity: "
        f"{overall_subgroups['nondemented_specificity']}"
    ),
    "",
    "DATA POLICY",
    "Old validation.csv: NOT LOADED",
    "Old test.csv:       NOT LOADED",
    "",
]

SUMMARY_TXT.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print()
print("=" * 96)
print("V3 RESNET18 BASELINE COMPLETE")
print("=" * 96)
print()
print("OOF metrics @ threshold 0.5")
print(
    f"Accuracy:          "
    f"{overall_metrics['accuracy']:.6f}"
)
print(
    f"Balanced Accuracy: "
    f"{overall_metrics['balanced_accuracy']:.6f}"
)
print(
    f"Sensitivity:       "
    f"{overall_metrics['sensitivity']:.6f}"
)
print(
    f"Specificity:       "
    f"{overall_metrics['specificity']:.6f}"
)
print(
    f"Precision:         "
    f"{overall_metrics['precision']:.6f}"
)
print(
    f"F1:                "
    f"{overall_metrics['f1']:.6f}"
)
print(
    f"ROC-AUC:           "
    f"{overall_metrics['roc_auc']:.6f}"
)
print(
    f"PR-AUC:            "
    f"{overall_metrics['pr_auc']:.6f}"
)
print(
    f"VeryMild Sens.:    "
    f"{overall_subgroups['verymild_sensitivity']}"
)
print(
    f"Mild Sens.:        "
    f"{overall_subgroups['mild_sensitivity']}"
)
print(
    f"NonDemented Spec.: "
    f"{overall_subgroups['nondemented_specificity']}"
)
print()
print("Old validation/test were NOT loaded.")
print()
print("Saved:")
print(OOF_FILE)
print(FOLD_METRICS_FILE)
print(SUMMARY_JSON)
print(SUMMARY_TXT)
print(history_file)
