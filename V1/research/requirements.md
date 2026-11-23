# Clinical Requirements for Chest X-Ray MVP

## Objective
Train a binary/multi-class classifier on the `COVID19_XRay` dataset capable of running efficiently in a clinical setting.

## Requirements

1. **Hardware Efficiency**
   - The model must fit within an 8GB VRAM footprint during training.
   - Inference must take < 500ms per image on CPU.

2. **Explainability**
   - The model MUST output a Grad-CAM heatmap highlighting the lung opacities/consolidation regions that led to the positive classification.
   - Black-box predictions are unacceptable in a clinical workflow.

3. **Input Format**
   - Images must be processed at $224 \times 224$ minimum resolution.
   - Must handle single-channel (grayscale) or 3-channel (RGB replicated) equally.

4. **False Negative Tolerance**
   - False negatives (missing a disease) are highly penalized. The loss function should ideally be weighted or recall must be prioritized during model selection.
