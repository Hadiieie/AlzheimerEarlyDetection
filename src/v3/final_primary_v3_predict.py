from pathlib import Path
import argparse
import hashlib
import json
import sys

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


# ============================================================
# FINAL PRIMARY V3 MRI PREDICTOR
#
# Primary project model:
# Normal vs Early_Demented
#
# Research / graduation-project use only.
# Not a clinical diagnosis.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FINAL_V3_DIR = (
    PROJECT_ROOT
    / "results"
    / "FINAL_MODEL_V3"
)

MODEL_FILE = (
    FINAL_V3_DIR
    / "final_v3_resnet18.pt"
)

CONFIG_FILE = (
    FINAL_V3_DIR
    / "final_v3_config.json"
)

FREEZE_FILE = (
    FINAL_V3_DIR
    / "freeze_manifest_v3.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "FINAL_PRIMARY_V3_PREDICTIONS"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DEVICE = torch.device("cpu")


# ============================================================
# HASH
# ============================================================

def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


# ============================================================
# PROJECT PREPROCESSING
# ============================================================

def resize_with_padding(
    image,
    target_size=220,
):

    h, w = image.shape

    scale = min(
        target_size / w,
        target_size / h,
    )

    new_w = max(
        1,
        int(round(w * scale)),
    )

    new_h = max(
        1,
        int(round(h * scale)),
    )

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros(
        (target_size, target_size),
        dtype=np.uint8,
    )

    x_offset = (
        target_size - new_w
    ) // 2

    y_offset = (
        target_size - new_h
    ) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w,
    ] = resized

    return canvas


def preprocess_raw_image(
    image_path,
):

    gray = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if gray is None:

        raise ValueError(
            f"Could not read image:\n{image_path}"
        )

    standardized = resize_with_padding(
        gray,
        target_size=220,
    )

    background_mask = (
        standardized == 0
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(
        standardized
    )

    filtered = cv2.medianBlur(
        enhanced,
        3,
    )

    filtered[
        background_mask
    ] = 0

    return filtered


def load_processed_image(
    image_path,
):

    gray = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if gray is None:

        raise ValueError(
            f"Could not read image:\n{image_path}"
        )

    return gray


# ============================================================
# CNN INPUT
# ============================================================

cnn_transform = transforms.Compose([

    transforms.Resize(
        (224, 224),
        antialias=True,
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
# MODEL
# ============================================================

def build_model():

    model = models.resnet18(
        weights=None
    )

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


@torch.no_grad()
def predict_probability(
    model,
    processed,
):

    image = (
        Image.fromarray(
            processed
        )
        .convert("RGB")
    )

    tensor = (
        cnn_transform(
            image
        )
        .unsqueeze(0)
        .to(DEVICE)
    )

    logits = model(
        tensor
    )

    probabilities = torch.softmax(
        logits,
        dim=1,
    )[0]

    normal_probability = float(
        probabilities[0]
        .cpu()
        .item()
    )

    early_probability = float(
        probabilities[1]
        .cpu()
        .item()
    )

    return (
        normal_probability,
        early_probability,
    )


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Final Primary V3 Alzheimer MRI Predictor"
    )
)

parser.add_argument(
    "image",
    type=str,
    help="Path to the MRI image.",
)

parser.add_argument(
    "--mode",
    choices=[
        "raw",
        "processed",
    ],
    default="raw",
    help=(
        "raw = apply full project preprocessing; "
        "processed = image is already a V3 processed image."
    ),
)

args = parser.parse_args()


# ============================================================
# START
# ============================================================

print()
print("=" * 72)
print("FINAL PRIMARY V3 MRI PREDICTOR")
print("=" * 72)
print()

image_path = Path(
    args.image
)

if not image_path.exists():

    raise FileNotFoundError(
        image_path
    )


# ============================================================
# VERIFY FINAL MODEL FILES
# ============================================================

for path in [
    MODEL_FILE,
    CONFIG_FILE,
    FREEZE_FILE,
]:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


with open(
    CONFIG_FILE,
    "r",
    encoding="utf-8",
) as f:

    config = json.load(
        f
    )


with open(
    FREEZE_FILE,
    "r",
    encoding="utf-8",
) as f:

    freeze = json.load(
        f
    )


threshold = float(
    config[
        "selected_threshold"
    ]
)


freeze_threshold = float(
    freeze[
        "selected_threshold"
    ]
)


if not np.isclose(
    threshold,
    freeze_threshold,
    atol=1e-15,
    rtol=0.0,
):

    raise RuntimeError(
        "Threshold mismatch between "
        "V3 config and freeze manifest."
    )


# ============================================================
# SHA256 VERIFICATION
# ============================================================

expected_hash = (
    freeze
    .get(
        "sha256",
        {},
    )
    .get(
        MODEL_FILE.name
    )
)


actual_hash = sha256_file(
    MODEL_FILE
)


if (
    expected_hash is not None
    and
    actual_hash != expected_hash
):

    raise RuntimeError(
        "FINAL V3 MODEL SHA256 MISMATCH."
    )


print(
    "Frozen V3 integrity: PASS"
)

print(
    f"Decision threshold: "
    f"{threshold:.12f}"
)

print()


# ============================================================
# LOAD MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=False,
)


checkpoint_threshold = float(
    checkpoint[
        "selected_threshold"
    ]
)


if not np.isclose(
    checkpoint_threshold,
    threshold,
    atol=1e-15,
    rtol=0.0,
):

    raise RuntimeError(
        "Checkpoint threshold mismatch."
    )


model = build_model()


model.load_state_dict(
    checkpoint[
        "state_dict"
    ]
)


model.eval()


# ============================================================
# PREPROCESS INPUT
# ============================================================

if args.mode == "raw":

    processed = preprocess_raw_image(
        image_path
    )

else:

    processed = load_processed_image(
        image_path
    )


if processed.size == 0:

    raise RuntimeError(
        "Processed image is empty."
    )


if np.isnan(
    processed.astype(
        np.float32
    )
).any():

    raise RuntimeError(
        "NaN values detected."
    )


preview_path = (
    OUTPUT_DIR
    / (
        image_path.stem
        + "_v3_preprocessed.png"
    )
)


cv2.imwrite(
    str(
        preview_path
    ),
    processed,
)


# ============================================================
# PREDICTION
# ============================================================

(
    probability_normal,
    probability_early,
) = predict_probability(
    model,
    processed,
)


prediction = int(
    probability_early
    >= threshold
)


if prediction == 1:

    prediction_name = (
        "Early_Demented"
    )

    model_score = (
        probability_early
    )

else:

    prediction_name = (
        "Normal"
    )

    model_score = (
        probability_normal
    )


# ============================================================
# RESULT
# ============================================================

print("=" * 72)
print("PREDICTION RESULT")
print("=" * 72)
print()

print(
    f"Input image:"
)

print(
    image_path
)

print()

print(
    f"Prediction: "
    f"{prediction_name}"
)

print()

print(
    f"Model score: "
    f"{model_score * 100:.2f}%"
)

print()

print(
    f"Normal probability: "
    f"{probability_normal * 100:.2f}%"
)

print(
    f"Early Demented probability: "
    f"{probability_early * 100:.2f}%"
)

print()

print(
    f"Decision threshold: "
    f"{threshold:.6f}"
)

print()

print(
    "Primary model: V3 ResNet18"
)

print(
    "Research-use prediction only."
)

print(
    "This output is NOT a clinical diagnosis."
)


# ============================================================
# SAVE RESULT
# ============================================================

result = {

    "input_image":
        str(
            image_path
        ),

    "input_mode":
        args.mode,

    "primary_model":
        "V3 ResNet18",

    "task":
        "Normal vs Early_Demented",

    "prediction":
        prediction_name,

    "model_score":
        float(
            model_score
        ),

    "model_score_percent":
        float(
            model_score
            * 100.0
        ),

    "normal_probability":
        probability_normal,

    "early_demented_probability":
        probability_early,

    "decision_threshold":
        threshold,

    "preprocessed_preview":
        str(
            preview_path
        ),

    "model_sha256":
        actual_hash,

    "research_use_only":
        True,

    "clinical_diagnosis":
        False,
}


result_path = (
    OUTPUT_DIR
    / (
        image_path.stem
        + "_v3_prediction.json"
    )
)


with open(
    result_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        result,
        f,
        indent=4,
    )


print()
print("=" * 72)
print("SAVED")
print("=" * 72)

print(
    result_path
)

print(
    preview_path
)