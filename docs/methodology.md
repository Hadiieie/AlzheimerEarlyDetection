# Methodology

## Classification Task

The primary task is binary MRI classification:

- Normal: NonDemented
- Early Demented: VeryMildDemented + MildDemented

ModerateDemented was excluded from the primary early-detection task.

## Dataset

The original dataset contained 39,538 MRI images.

The primary binary task contained 29,853 images.

## Leakage Control

The project used leakage-group analysis and group-aware splitting.

Primary split:

- Training: 20,891
- Validation: 4,482
- Test: 4,480

No leakage-group overlap was permitted between the splits.

## Image Preprocessing

The primary preprocessing pipeline consisted of:

1. Grayscale conversion
2. Aspect-preserving resize
3. Padding to 220 x 220 pixels
4. CLAHE contrast enhancement
5. 3 x 3 median filtering
6. Conversion to three-channel input for ResNet18
7. Resize to 224 x 224
8. ImageNet normalization

## Model Development

### V1

RBF SVM using 30 engineered MRI features.

### V2

Optimized RBF SVM using group-aware cross-validation.

### V3

ImageNet-pretrained ResNet18 using group-aware three-fold
out-of-fold evaluation.

The training recipe used:

- classifier-head training
- layer4 fine-tuning
- AdamW
- class weighting
- controlled augmentation

## Final V3 Performance

- Accuracy: 92.633%
- Balanced Accuracy: 91.994%
- Sensitivity: 93.923%
- Specificity: 90.066%
- Precision: 94.954%
- F1: 94.436%
- ROC-AUC: 97.785%
- PR-AUC: 98.839%
- MCC: 0.8355

These results represent leakage-safe group-aware out-of-fold
development performance.

They are not independent external clinical validation.

## Cross-Dataset Evaluation

A separate OASIS challenge study was performed to investigate
cross-dataset generalization.

The experiment revealed substantial domain shift between the primary
MRI dataset and OASIS.

This study is therefore treated as a generalization analysis rather
than the headline model-performance result.

## Research Scope

This project is intended for academic and research purposes only and
is not a clinical diagnostic system.