## III. Methodology

### 3.1 Modality-Aware Tokenization
To ingest multi-modality volumetric ($3\text{D}$) and projection ($2\text{D}$) medical data into a single transformer backbone, we formulate a **Modality-Aware Tokenizer**. Let the input medical image volume be represented as $X \in \mathbb{R}^{H \times W \times D \times C}$, where $H, W, D$ represent height, width, and depth, respectively, and $C$ represents the input channel count (typically $C=1$ for grayscale medical data). For projection images (e.g., X-Rays), we set $D=1$.

We extract non-overlapping volumetric patches of size $P_H \times P_W \times P_D$. These patches are projected to a $D$-dimensional latent space using a learnable linear projection $W_E$. To retain structural physical parameters, we append a **Modality-Aware Meta-Embedding** ($E_{\text{meta}}$) to each token, representing physical acquisition parameters:

$$E_{\text{meta}} = \text{MLP}\left([ \Delta_x, \Delta_y, \Delta_z, \mathbf{m} ]\right)$$

where $\Delta_x, \Delta_y, \Delta_z$ represent the spatial voxel spacing in millimeters, and $\mathbf{m}$ is a one-hot vector indicating the scanner modality (e.g., CT, MRI, Ultrasound, X-Ray). The final patch representation $z_i$ is defined as:

$$z_i = (x_i \cdot W_E) + E_{\text{pos}} + E_{\text{meta}}$$

where $E_{\text{pos}}$ represents learnable 3D positional embeddings.

### 3.2 Saliency-Guided Masking (SGM)
Standard Masked Autoencoders randomly mask a high percentage (typically $75\% - 85\%$) of patches. In multi-organ medical imaging, this approach leads to **dominant-organ shortcutting**, where the model reconstructs low-entropy tissues (such as homogenous muscle or fat) without capturing complex anatomical features of smaller organs or lesions.

To prevent this, we introduce **Saliency-Guided Masking (SGM)**. For each volume, we compute a baseline saliency map $S \in \mathbb{R}^{H \times W \times D}$ representing localized anatomical entropy or intensity gradients:

$$S(x,y,z) = \nabla(X(x,y,z))$$

We calculate a saliency score $s_i$ for each patch $i$ by averaging the saliency values within the patch volume. The probability of masking patch $i$, denoted by $P(\text{mask}_i)$, is inversely proportional to its saliency score:

$$P(\text{mask}_i) = \frac{\exp(-s_i / \tau)}{\sum_j \exp(-s_j / \tau)}$$

where $\tau$ is a temperature parameter controlling the uniformity of the masking. SGM forces the network to retain high-information tokens (e.g., boundary transitions, fine vessels, and lesion margins) during encoding, forcing the reconstruction decoder to solve a more challenging anatomical synthesis problem.

### 3.3 Reconstruction Pretext Task & Decoupled Heads
The backbone is optimized using a dual-objective loss function combining a structural reconstruction loss ($\mathcal{L}_{\text{recon}}$) and a contrastive semantic alignment loss ($\mathcal{L}_{\text{align}}$):

$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{recon}} + (1 - \alpha) \mathcal{L}_{\text{align}}$$

To bridge the actionability gap, we bypass task-specific classification decoders. Instead, the latent representation is decoded through an **Action Planning Head** that outputs diagnostic action trajectories (e.g., suggesting secondary staging scans, recommending needle biopsy targets, or estimating regional hazard scores for survival predictions).
