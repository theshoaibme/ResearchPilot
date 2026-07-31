# Phase 4: Key Gaps Finding & Analysis

---

Synthesizing literature review insights and clinical requirements reveals **three primary gaps**:

## Gap 1: Dominant-Organ Shortcutting in Masked Autoencoders (MAE)
- **Problem:** Standard random masking (e.g. 75–85% patch dropping) allows models to reconstruct overall volume by learning low-entropy shortcuts (homogeneous bone or abdominal cavity fat), ignoring fine boundaries or early-stage lesions.
- **Impact:** High reconstruction metrics on trivial regions, but weak performance on delicate multi-organ pathologies.

---

## Gap 2: Missing Not at Random (MNAR) Vulnerability
- **Problem:** Real-world diagnostic scans rarely cover the entire torso. A patient may receive an isolated liver MRI or chest CT. Standard foundation models suffer severe accuracy decay when processing truncated/missing body regions.
- **Impact:** Up to 25.3% drop in segmentation accuracy when input coverage is partial.

---

## Gap 3: Actionability & Transparency Disconnect
- **Problem:** Existing AI models generate black-box probability distributions or basic diagnostic classes without explaining *why* or guiding *what clinical step to take next*.
- **Impact:** Radiologists distrust outputs, creating a barrier to clinical adoption.
