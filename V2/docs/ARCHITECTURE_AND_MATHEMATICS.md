# Architecture & Mathematical Specification: Pan-Organ Net V2

## 1. Design Objectives & SOTA Positioning

Contemporary medical foundation models demonstrate strong capabilities in isolated regimes—such as 2D text-prompted parsing (*BiomedParse*, Zhao et al., *Nature Methods* 2025), 3D text-prompted segmentation (*SAT*, Zhao et al., 2024), 3D CT segmentation (*VISTA3D*, He et al., 2024; *TotalSegmentator*, Wasserthal et al., 2023), and multi-task MoE vision-language connectors (*Uni-Med*, Zhu et al., *NeurIPS* 2024). However, three structural bottlenecks persist when unifying **2D projection modalities** (X-Ray, Ultrasound, Mammography) and **volumetric 3D modalities** (CT, MRI) inside a single deployable clinical model:

1. **Dimensionality Collapse & Modality Physics Mismatch:** Feeding 2D scans ($D=1$) directly into naive 3D volumetric pooling layers causes depth collapse, while uniform min-max scaling destroys physical CT Hounsfield Units and MRI tissue contrast.
2. **Cross-Modality & Multi-Task Gradient Conflict:** Joint optimization across 2D thoracic multi-label classification and 3D abdominal multi-organ segmentation induces gradient interference ("tug-of-war") that degrades specialist accuracy.
3. **Partial Field-of-View (MNAR) Degradation & Uncalibrated Uncertainty:** Clinical scans frequently cover truncated anatomical slabs; models relying on rigid boundary coordinates degrade sharply under partial FOV, while uncalibrated softmax heads emit overconfident predictions under domain shift.

**Pan-Organ Net V2** resolves all three limitations inside a single parameter-efficient architecture designed to train reproducibly on single-GPU academic compute.

---

## 2. Module-by-Module Mathematical Formulation

### 2.1 Omni-Modal Stem (`OmniModalityStem`) & Physics-Informed FiLM

Given a clinical input $X$ with dimensionality indicator $\mathbb{I}_{3D} \in \{0, 1\}$, 2D projections ($X_{2D} \in \mathbb{R}^{3 \times H \times W}$) and 2.5D volumetric slabs ($X_{2.5D} \in \mathbb{R}^{K \times H \times W}$, $K=5$ adjacent axial slices) are projected through dedicated convolutional stems $f_{2D}$ and $f_{2.5D}$ into a shared spatial feature map $F_0 \in \mathbb{R}^{C_s \times H \times W}$:

$$F_0 = \mathbb{I}_{3D} \cdot f_{2.5D}(X) + (1 - \mathbb{I}_{3D}) \cdot f_{2D}(X_{1:3})$$

Physical acquisition metadata is encoded as a continuous 10-dimensional vector:

$$\mathbf{u} = \left[\log \Delta_x,\; \log \Delta_y,\; \log \Delta_z,\; \mathbb{I}_{3D},\; c_{\text{contrast}},\; \mathbf{m}_{\text{one-hot}}\right] \in \mathbb{R}^{10}$$

where $(\Delta_x, \Delta_y, \Delta_z)$ are physical voxel spacings in millimeters and $\mathbf{m}_{\text{one-hot}} \in \{0,1\}^5$ denotes modality (CT, MRI, CXR, US, MG). Rather than additive bias injection, metadata modulates features via **Feature-wise Linear Modulation (FiLM)** initialized at identity ($\gamma = \mathbf{0}, \beta = \mathbf{0}$):

$$\tilde{F}_0 = \left(1 + \gamma(\mathbf{u})\right) \odot F_0 + \beta(\mathbf{u})$$

---

### 2.2 Modality-Adaptive Uncertainty-Guided Masking (`MA-UGM`)

During Stage-1 Self-Supervised Pretraining, naive gradient-magnitude masking ($\|\nabla X\|$) fails on Ultrasound (where acoustic speckle generates uniformly high gradients) and MRI (where bias fields distort raw intensity gradients). `MA-UGM` replaces raw gradients with a two-phase modality-adaptive importance score $s_i \in [0, 1]$ per spatial patch $i$:

1. **Warmup Phase (Modality-Specific Foreground Saliency):**
   - **Ultrasound:** Median-despeckled ($5 \times 5$) Sobel structural gradient masked by acoustic foreground.
   - **MRI:** Foreground-masked local intensity standard deviation ($7 \times 7$ window).
   - **CT / CXR / MG:** Body-foreground-masked Gaussian-smoothed structural gradient.
2. **Online Self-Distillation Phase (EMA Teacher Residual Uncertainty):**
   An Exponential Moving Average (EMA) teacher ($\theta_{\text{EMA}} \leftarrow 0.996\,\theta_{\text{EMA}} + 0.004\,\theta$) computes patch-level reconstruction residuals $e_i$. Combined patch importance is $c_i = (1 - \eta) s_i + \eta e_i$, yielding inverse-saliency masking probabilities:

$$P(\text{mask}_i) = \frac{\exp(-c_i / \tau)}{\sum_{j=1}^{N} \exp(-c_j / \tau)}$$

High-information anatomical boundaries and high-uncertainty lesion regions are preferentially retained in the visible encoder context, preventing background/air shortcut reconstruction.

---

### 2.3 Hierarchical Backbone with LoRA & Sparse Top-2 `MT-MoE`

A pretrained 4-stage hierarchical ConvNeXt-Small backbone extracts multi-scale representations $\{F_1, F_2, F_3, F_4\}$, adapted via **Low-Rank Adaptation (LoRA)** ($W = W_0 + \frac{\alpha}{r} B A$, $r=8, \alpha=16$) and stage-wise `PhysicsFiLM`.

To eliminate multi-modality and multi-task gradient conflict, stages 3 and 4 incorporate **Sparse Top-2 Modality-Task Mixture-of-Experts (`SparseMTMoELayer`)** with $E=4$ depthwise-separable spatial experts $\{g_1, \dots, g_E\}$. A gating network conditioned on global pooled visual context and acquisition metadata computes routing probabilities $\mathbf{p} = \text{softmax}(W_r [\text{GAP}(F_l); \mathbf{u}])$ and activates only the top $k=2$ experts per sample:

$$F_l^{\text{MoE}} = \text{GroupNorm}\left(F_l + \sum_{e \in \text{TopK}(\mathbf{p}, 2)} \tilde{p}_e \, g_e(F_l)\right)$$

Expert collapse is prevented via the Switch-style load-balancing auxiliary objective:

$$\mathcal{L}_{\text{bal}} = E \sum_{e=1}^{E} f_e \cdot \bar{p}_e$$

---

### 2.4 Self-Supervised Anatomical Coordinate Regressor (`SACR`)

To ensure robustness under **Missing Not at Random (MNAR)** partial-FOV scan truncation without circularly assuming any fixed landmark (such as the tracheal carina) appears inside a cropped scan, `SACR` regresses continuous normalized body-atlas coordinates $\hat{\mathbf{c}} = (\hat{z}_{\text{inf-sup}}, \hat{y}_{\text{ap}}, \hat{x}_{\text{lr}}) \in [0, 1]^3$ directly from $F_4$:

$$\mathcal{L}_{\text{SACR}} = \text{SmoothL1}(\hat{\mathbf{c}}, \mathbf{c}_{\text{true}})$$

During training, random contiguous 20%–50% FOV slab truncations teach the encoder to infer global anatomical location from local organ texture and feed the predicted coordinate embedding back into the mask decoder.

---

### 2.5 Prompt-Conditioned Multi-Task Decoders & Composite Clinical Losses

Standardized SNOMED/RadLex task prompts (e.g., *"segment abdominal organs in 3D computed tomography"*) are encoded via a frozen biomedical text encoder into query vector $\mathbf{q} \in \mathbb{R}^{384}$, conditioning two unified heads:

1. **Universal Mask Decoder (FPN):** Fuses $\{F_1, F_2, F_3, F_4\}$ top-down, modulates via prompt query $\mathbf{q}$, and optimizes a composite **Dice + Focal Cross-Entropy + Boundary Sobel L1 Loss**:
   $$\mathcal{L}_{\text{seg}} = \mathcal{L}_{\text{SoftDice}} + \mathcal{L}_{\text{FocalCE}}(\gamma=2) + 0.3\,\mathcal{L}_{\text{BoundaryL1}}$$
2. **Clinical Pathology Head:** Fuses pooled $F_4$ with $\mathbf{q}$ to drive multi-label thoracic screening via **Asymmetric Focal Loss** ($\gamma_+ = 0, \gamma_- = 3$, prioritizing recall on rare/severe clinical findings) and multi-class lesion grading via **Focal Cross-Entropy**.

The unified Stage-2 training objective is:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{seg}} \mathcal{L}_{\text{seg}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_{\text{SACR}} \mathcal{L}_{\text{SACR}} + \lambda_{\text{bal}} \mathcal{L}_{\text{bal}}$$

---

## 3. Modality-Conditional Conformal Prediction (`MC2-CP`)

Post-hoc **Temperature Scaling** calibrates softmax logits on the validation split ($T^* = \arg\min_T \text{NLL}(\mathbf{z}/T, y)$). Subsequently, `MC2-CP` stratifies nonconformity scores $s_i = 1 - \hat{p}_{y_i}(x_i)$ by clinical group $g \in \{\text{modality} \times \text{coverage\_state}\}$ and computes finite-sample Mondrian quantiles:

$$\hat{q}_g = \text{Quantile}\left(\{s_i : g_i = g\},\; \frac{\lceil (n_g + 1)(1 - \alpha) \rceil}{n_g}\right)$$

At inference, the calibrated prediction set $\mathcal{C}(x) = \{y : 1 - \hat{p}_y(x) \le \hat{q}_g\}$ guarantees marginal and per-modality coverage $\ge 1 - \alpha$ (90% for $\alpha=0.10$), flagging ambiguous cases ($|\mathcal{C}(x)| > 1$) for mandatory specialist review.
