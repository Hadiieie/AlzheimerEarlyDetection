# Trained Model

The final primary model is the frozen **V3 ResNet18** classifier.

The trained PyTorch checkpoint is not stored directly in this repository
because model binaries are excluded from the main source repository.

Primary model:

- Architecture: ResNet18
- Pretraining: ImageNet
- Task: Normal vs Early Demented
- Decision threshold: approximately 0.48834

The model configuration was frozen before subsequent challenge-set
experiments.

For research and academic use only.