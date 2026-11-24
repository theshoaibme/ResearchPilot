window.PANORGAN_REPORTS = {
  "1": {
    id: "1",
    title: "PanOrgan 1",
    subtitle: "Technical Architecture & Prototype Report",
    stage: "Version 1 (Foundational)",
    scope: "Architecture & Data Staging",
    status: "Completed",
    overview: [
      "PanOrgan 1 established the foundational research scope and architectural blueprint of the PanOrgan project. Its primary objective was to eliminate clinical model fragmentation by designing a unified multi-modal, multi-task medical foundation framework.",
      "The design targeted volumetric (3D CT/MRI), projection (2D X-Ray), optical (Endoscopy), and acoustic (Ultrasound) inputs under a single backbone with ontology-aligned clinical reporting."
    ],
    mermaid: `flowchart LR
      A["<b>Clinical Inputs</b><br/>3D CT/MRI, 2D CXR, US"]:::input --> B["<b>HAT Tokenizer</b><br/>Hierarchical Spatial Tokens"]:::core
      B --> C["<b>SGM Pretraining</b><br/>Saliency-Guided MAE"]:::train
      C --> D["<b>PanOrgan 1</b><br/>7 Clinical Domains"]:::out
      classDef input fill:#eff6ff,stroke:#2563eb,color:#1e3a8a,stroke-width:1.5px;
      classDef core fill:#f0fdf4,stroke:#16a34a,color:#14532d,stroke-width:1.5px;
      classDef train fill:#fefce8,stroke:#ca8a04,color:#713f12,stroke-width:1.5px;
      classDef out fill:#faf5ff,stroke:#9333ea,color:#581c87,stroke-width:1.5px;`,
    architectureRows: [
      ["Hierarchical Anatomical Tokenizer (HAT)", "Maps heterogeneous 2D and 3D imaging modalities into a unified spatial token representation.", "Specified in proposal; refined into OmniModalityStem in Version 2."],
      ["Saliency-Guided Masked Autoencoder (SGM)", "Self-supervised pretraining objective designed to preserve small lesion and organ boundaries.", "Formulated in design doc; operationalized as Stage 1 SSL in Version 3."],
      ["Clinical Ontology Alignment", "Structured mapping across 7 clinical domains (Urology, Pulmonology, Cardiology, Hepatology, Nephrology, Gastroenterology, Musculoskeletal).", "Implemented via OntologyPromptEncoder in Version 2+."]
    ],
    datasetsNote: "PanOrgan 1 defined a 3-Tier Data Allocation Strategy separating Tier 1 (Pretraining), Tier 2 (Supervised Multi-Task), and Tier 3 (External Validation) across initial prototype cohorts.",
    datasetHeaders: ["Tier Allocation", "Target Modalities", "Staging Role", "Clinical Scope"],
    datasetRows: [
      ["Tier 1: Self-Supervised SSL", "3D CT/MRI, 2D CXR, US", "Unlabeled Anatomical Pretraining", "Multi-organ masked representation learning"],
      ["Tier 2: Supervised Multi-Task", "MRI, CXR, Ultrasound", "Joint Segmentation & Classification", "Oncology, Pulmonology, Breast & Nerve imaging"],
      ["Tier 3: External Validation", "Multi-Modal Clinical Cohorts", "Out-of-Distribution Generalization", "Cross-scanner & MNAR missingness evaluation"]
    ],
    trainingNote: "PanOrgan 1 served as the architectural specification and initial baseline prototype phase. Unified multi-cohort joint training and conformal calibration began in PanOrgan 2.",
    epochHeaders: null,
    epochRows: null,
    classificationRows: null,
    segmentationRows: null,
    conformalRows: null,
    images: [
      { src: "assets/v1_workflow.png", caption: "Figure 1: PanOrgan 1 Prototype Workflow Diagram" },
      { src: "assets/v1_confusion_matrix.png", caption: "Figure 2: PanOrgan 1 Prototype Confusion Matrix" },
      { src: "assets/v1_train_val_acc.png", caption: "Figure 3: PanOrgan 1 Training & Validation Accuracy" },
      { src: "assets/v1_train_val_loss.png", caption: "Figure 4: PanOrgan 1 Training & Validation Loss" },
      { src: "assets/v1_train_val_prec.png", caption: "Figure 5: PanOrgan 1 Training & Validation Precision" },
      { src: "assets/v1_train_val_rec.png", caption: "Figure 6: PanOrgan 1 Training & Validation Recall" }
    ],
    svgBars: null,
    verificationRows: [
      ["Master Proposal & Architecture", "297-line Academic Specification", "Completed & Verified"],
      ["Prototype Core & Dataset Archive", "External SSD Archive", "Verified (Migrated off internal disk)"],
      ["Multi-Cohort Unified Benchmark", "Pre-benchmark prototype stage", "Introduced in PanOrgan 2"]
    ],
    robustnessNote: "PanOrgan 1 is a pre-benchmark architectural and prototype staging release. Held-out multi-cohort evaluation (Dice, AUROC, Macro-F1, ECE) began in PanOrgan 2.",
    limitations: [
      ["No Unified Multi-Cohort Benchmark", "Solved in PanOrgan 2 via end-to-end held-out test evaluation (n = 825 cls / 646 seg)."],
      ["Host Internal Storage Bottleneck", "Solved in PanOrgan 2 by migrating datasets, shards, and caches to external SSD."],
      ["No Automated Patient-Level Split Audit", "Solved in PanOrgan 2 with strict patient_id zero-leakage auditing (leakage_count = 0)."],
      ["No Uncertainty Calibration", "Solved in PanOrgan 2 with temperature scaling and MC2 conformal prediction."]
    ]
  },

  "2": {
    id: "2",
    title: "PanOrgan 2",
    subtitle: "Technical Implementation & Benchmark Report",
    stage: "Version 2 (3-Cohort Baseline)",
    scope: "MRI, CXR, US (n = 825 Test)",
    status: "Completed & Evaluated",
    overview: [
      "PanOrgan 2 transitioned the project from architectural design into an executable, reproducible medical AI pipeline. It introduced patient-level zero-leakage sharding, external SSD storage routing, multi-stage training on Apple M4 (MPS), temperature scaling, and conformal prediction evaluation across three core cohorts."
    ],
    mermaid: `flowchart LR
      A["<b>3 Verified Cohorts</b><br/>TCGA-LGG, BUSI, NIH-CXR14"]:::input --> B["<b>Zero-Leakage Shards</b><br/>0 Leaked Patients"]:::audit
      B --> C["<b>PanOrgan 2</b><br/>OmniModalityStem +<br/>OntologyPromptEncoder"]:::core
      C --> D["<b>Joint Seg & Cls</b><br/>Dice 0.7790 | AUROC 0.9062"]:::train
      D --> E["<b>MC2 Conformal</b><br/>T=1.6201 | Cov 91.03%"]:::out
      classDef input fill:#eff6ff,stroke:#2563eb,color:#1e3a8a,stroke-width:1.5px;
      classDef audit fill:#ecfdf5,stroke:#059669,color:#064e3b,stroke-width:1.5px;
      classDef core fill:#eef2ff,stroke:#4f46e5,color:#312e81,stroke-width:1.5px;
      classDef train fill:#fefce8,stroke:#ca8a04,color:#713f12,stroke-width:1.5px;
      classDef out fill:#faf5ff,stroke:#9333ea,color:#581c87,stroke-width:1.5px;`,
    architectureRows: [
      ["OmniModalityStem", "Standardized heterogeneous MRI, CXR, and US inputs into consistent spatial tokens.", "Implemented & evaluated."],
      ["OntologyPromptEncoder", "Structured clinical prompt embeddings for organ and modality conditioning.", "Implemented & evaluated."],
      ["Calibration & Conformal Head", "Post-hoc temperature scaling (T = 1.6201) and Marginal/Mondrian Conformal Calibration (MC2).", "Target 90% coverage achieved (91.03%)."]
    ],
    datasetsNote: "PanOrgan 2 trained and evaluated on 3 verified cohorts with strict patient-level zero-leakage splitting: n = 825 classification test samples and n = 646 segmentation test samples.",
    datasetRows: [
      ["tcga_lgg", "MRI", "537 (test)", "Joint segmentation & classification"],
      ["nih_cxr14", "CXR", "179 (test)", "Multi-label thoracic classification"],
      ["busi", "US", "109 (test)", "Breast ultrasound segmentation & classification"]
    ],
    trainingNote: "Trained on Apple M4 GPU (MPS) with external SSD sharding (15 SSL epochs reaching val_loss = 0.0061, followed by 10 Multi-Task epochs reaching best val_loss = 0.8941 at Epoch 3). Post-hoc temperature scaling optimized T = 1.6201 on validation logits, followed by MC2 conformal calibration targeting 90% marginal coverage (alpha = 0.10).",
    epochHeaders: ["Epoch", "Train Loss", "Val Loss", "Seg Loss", "Cls Loss", "SACR Loss"],
    epochRows: [
      ["1", "1.2687", "0.9263", "1.0519", "0.1961", "0.0015"],
      ["2", "1.1232", "0.9269", "1.0005", "0.1021", "0.0012"],
      ["3 (Best Val)", "1.0783", "0.8941", "0.9905", "0.0672", "0.0011"],
      ["5", "1.0391", "0.8963", "0.9779", "0.0408", "0.0010"],
      ["8", "1.0011", "0.9188", "0.9631", "0.0176", "0.0008"],
      ["10 (Final)", "0.9898", "0.9592", "0.9590", "0.0105", "0.0007"]
    ],
    classificationRows: [
      ["AUROC", "0.9062", "0.9002"],
      ["Accuracy", "69.82%", "69.82%"],
      ["Macro-F1", "0.6011", "0.6011"],
      ["Macro Precision", "0.6020", "0.6020"],
      ["Macro Sensitivity (Recall)", "0.7838", "0.7838"],
      ["Macro Specificity", "0.8663", "0.8663"],
      ["ECE", "0.1296", "0.1609"],
      ["Brier Score", "0.4304", "0.4208"]
    ],
    segmentationRows: [
      ["tcga_lgg", "537", "0.8081 (0.7829–0.8331)", "0.7653", "10.62"],
      ["busi", "109", "0.6356 (0.5805–0.6855)", "0.5211", "53.11"],
      ["Overall", "646", "0.7790 (0.7548–0.8034)", "0.7241", "17.74"]
    ],
    conformalRows: [
      ["tcga_lgg", "537", "92.18%", "1.05", "4.66%"],
      ["busi", "109", "91.74%", "1.29", "29.36%"],
      ["nih_cxr14", "179", "87.15%", "2.87", "100.00%"],
      ["Overall", "825", "91.03%", "1.48", "28.61%"]
    ],
    images: [
      { src: "assets/v2_confusion_matrix.png", caption: "Figure 1: PanOrgan 2 Test Confusion Matrix (n = 825)" },
      { src: "assets/v2_train_val_loss.png", caption: "Figure 2: PanOrgan 2 Training & Validation Loss (10 Epochs)" },
      { src: "assets/v2_classification_metrics.png", caption: "Figure 3: PanOrgan 2 Test Accuracy, Precision, Recall & F1 (n = 825)" },
      { src: "assets/v2_segmentation_dice.png", caption: "Figure 4: PanOrgan 2 Segmentation Dice & IoU by Cohort (n = 646)" },
      { src: "assets/v2_conformal_coverage.png", caption: "Figure 5: PanOrgan 2 MC2 Conformal Coverage & Abstention" },
      { src: "assets/v2_mnar_stress.png", caption: "Figure 6: PanOrgan 2 MNAR Missingness Robustness Curve" },
      { src: "assets/v2_evaluation_plots.png", caption: "Figure 7: PanOrgan 2 Combined Evaluation & Calibration Curves" }
    ],
    svgBars: null,
    verificationRows: [
      ["Zero-Leakage Split Audit", "Patient-level split verification across 3 cohorts", "Passed (0 leaked patients)"],
      ["Held-Out Test Evaluation", "Independent test split (825 cls / 646 seg)", "Verified (AUROC 0.9062, Dice 0.7790)"],
      ["Conformal Coverage (alpha=0.10)", "MC2 Marginal Coverage", "91.03% (avg set size 1.4752)"]
    ],
    robustnessNote: "MC2 Conformal Coverage: 91.03% marginal coverage, average set size 1.4752, abstention rate 28.61% (nih_cxr14 abstention 100%). MNAR Stress Curve (Overall Dice): 0.7790 (0%) -> 0.7886 (20%) -> 0.7589 (40%).",
    limitations: [
      ["Moderate Classification F1 (0.6011) & Accuracy (69.82%)", "Solved in PanOrgan 3 via 3-stage curriculum (SSL + Joint + Head Calibration), raising Macro-F1 to 0.8905 and Accuracy to 87.93%."],
      ["Temperature Scaling ECE Regression (0.1296 -> 0.1609)", "Solved in PanOrgan 3 (T = 0.9390), reducing calibrated ECE to 0.0598 and nih_cxr14 abstention from 100%."],
      ["High Ultrasound Boundary Error (BUSI HD95 = 53.11 mm)", "Improved in PanOrgan 3: BUSI Dice rose from 0.6356 to 0.7790 and HD95 dropped to 28.78 mm."],
      ["Limited Cohort & Modality Diversity (3 cohorts)", "Expanded to 7 supervised cohorts in PanOrgan 3 and all 5 modalities (adding CT & MG) in PanOrgan 3.1."]
    ]
  },

  "3": {
    id: "3",
    title: "PanOrgan 3",
    subtitle: "PanOrgan 3 Cohort — Multi-Stage SSL & Calibrated Release Report",
    stage: "Version 3 (7-Cohort Multi-Stage)",
    scope: "MRI, CXR, US (n = 1,425 Cls / 1,246 Seg Test)",
    status: "Completed & Evaluated",
    overview: [
      "PanOrgan 3 (PanOrgan 3 Cohort) scaled the framework from 3 cohorts to 7 supervised clinical cohorts (and 8 total sharded cohorts including prostatex_mri for SSL), introducing a three-stage training curriculum (Stage 1 SSL, Stage 2A Joint Multi-Task, Stage 2B Head Calibration) that raised Macro-F1 from 0.6011 to 0.8905 and reduced ECE from 0.1609 to 0.0598."
    ],
    mermaid: `flowchart LR
      A["<b>Stage 1: SSL</b><br/>val_loss = 0.003693"]:::input --> B["<b>Stage 2A: Joint</b><br/>val_dice=0.8350 | val_f1=0.9009"]:::core
      B --> C["<b>PanOrgan 3 Cohort</b><br/>Stage 2B Calibrated (240 MB)"]:::train
      C --> D["<b>Temp Scaling (T=0.9390)</b><br/>AUROC 0.9715 | ECE 0.0598"]:::audit
      D --> E["<b>MC2 Conformal</b><br/>Cov 93.33% | MNAR 0.7908"]:::out
      classDef input fill:#eff6ff,stroke:#2563eb,color:#1e3a8a,stroke-width:1.5px;
      classDef core fill:#eef2ff,stroke:#4f46e5,color:#312e81,stroke-width:1.5px;
      classDef train fill:#fefce8,stroke:#ca8a04,color:#713f12,stroke-width:1.5px;
      classDef audit fill:#ecfdf5,stroke:#059669,color:#064e3b,stroke-width:1.5px;
      classDef out fill:#faf5ff,stroke:#9333ea,color:#581c87,stroke-width:1.5px;`,
    architectureRows: [
      ["Unified Multimodal Backbone", "Shared encoder with OmniModalityStem and OntologyPromptEncoder across neurology, pulmonology, breast, and nerve cohorts.", "240 MB final calibrated weights; 715 MB Stage 2A joint weights."],
      ["Three-Stage Curriculum", "Separates masked anatomical SSL (Stage 1), joint seg+cls optimization (Stage 2A), and head calibration (Stage 2B).", "All 3 stages completed (~3.54 GB total checkpoints)."],
      ["Post-Hoc Calibration & Thresholds", "Temperature scaling (T = 0.9390) + High-Sens threshold tuning (0.6, 1.3, 1.3, 1.3).", "High-Sens operating point: Acc 88.07%, F1 0.8938, Sens 89.54%, Spec 94.61%."]
    ],
    datasetsNote: "Zero-leakage patient-level split across 8 sharded cohorts. Held-out test evaluation covered n = 1,425 classification samples and n = 1,246 segmentation samples.",
    datasetRows: [
      ["tcga_lgg", "MRI", "537 (test)", "Brain tumor joint segmentation & classification"],
      ["pulmonology_cxr", "CXR", "334 (test)", "Lung segmentation & pulmonary classification"],
      ["ultrasound_nerve", "US", "266 (test)", "Brachial plexus nerve segmentation & presence"],
      ["nih_cxr14", "CXR", "179 (test)", "Multi-label thoracic classification"],
      ["busi", "US", "109 (test)", "Breast ultrasound lesion segmentation & classification"]
    ],
    trainingNote: "Stage 1 SSL reached best_val_loss = 0.003693. Stage 2A (batch_size=8, grad_accum=4 on Apple M4 MPS) reached Ep 17 best_composite = 0.8679 (val_dice=0.8350, val_f1=0.9009, val_acc=88.90%). Stage 2B reached Ep 1 best_val_f1 = 0.9017, val_acc=89.18%.",
    epochHeaders: ["Stage / Epoch", "Train Loss", "Val Dice", "Val Macro-F1", "Val Accuracy", "Composite"],
    epochRows: [
      ["S2A Ep 1", "3.8687", "0.6524", "0.7201", "69.94%", "0.6863"],
      ["S2A Ep 5", "2.6548", "0.7933", "0.8575", "85.86%", "0.8254"],
      ["S2A Ep 9", "2.4102", "0.8253", "0.8924", "88.26%", "0.8588"],
      ["S2A Ep 14", "2.2754", "0.8363", "0.8966", "89.04%", "0.8665"],
      ["S2A Ep 17 (Best)", "2.2329", "0.8350", "0.9009", "88.90%", "0.8679"],
      ["S2B Ep 1 (Calib)", "Frozen BB", "0.8350", "0.9017", "89.18%", "0.8684"]
    ],
    classificationRows: [
      ["AUROC", "0.9715", "0.9713"],
      ["Accuracy", "87.93%", "87.93%"],
      ["Macro-F1", "0.8905", "0.8905"],
      ["Macro Precision", "0.8956", "0.8956"],
      ["Macro Sensitivity (Recall)", "0.8879", "0.8879"],
      ["Macro Specificity", "0.9442", "0.9442"],
      ["ECE", "0.0792", "0.0598"],
      ["Brier Score", "0.2105", "0.2077"]
    ],
    segmentationRows: [
      ["pulmonology_cxr", "334", "0.9637 (0.9600–0.9672)", "0.9316", "9.61"],
      ["tcga_lgg", "537", "0.8753 (0.8541–0.8951)", "0.8413", "6.90"],
      ["busi", "109", "0.7790 (0.7280–0.8240)", "0.6983", "28.78"],
      ["ultrasound_nerve", "266", "0.6175 (0.5620–0.6680)", "0.5863", "9.77"],
      ["Overall", "1,246", "0.8355 (0.8180–0.8510)", "0.7986", "10.16"]
    ],
    conformalRows: [
      ["nih_cxr14", "179", "100.00%", "1.00", "0.00%"],
      ["tcga_lgg", "537", "94.41%", "1.00", "0.00%"],
      ["pulmonology_cxr", "334", "88.92%", "1.00", "0.00%"],
      ["busi", "109", "93.58%", "1.57", "30.28%"],
      ["ultrasound_nerve", "266", "92.11%", "3.59", "100.00%"],
      ["Overall", "1,425", "93.33%", "1.53", "20.98%"]
    ],
    images: [
      { src: "assets/v3_confusion_matrix.png", caption: "Figure 1: PanOrgan 3 Test Confusion Matrix (n = 1,425)" },
      { src: "assets/v3_train_loss.png", caption: "Figure 2: PanOrgan 3 Stage 2A Training Loss (18 Epochs)" },
      { src: "assets/v3_val_accuracy_f1.png", caption: "Figure 3: PanOrgan 3 Validation Accuracy, Macro-F1 & Dice (18 Epochs)" },
      { src: "assets/v3_precision_recall_metrics.png", caption: "Figure 4: PanOrgan 3 Test Accuracy, Precision, Recall & F1 (n = 1,425)" },
      { src: "assets/v3_segmentation_dice.png", caption: "Figure 5: PanOrgan 3 Segmentation Dice & IoU by Cohort (n = 1,246)" },
      { src: "assets/v3_calibration_comparison.png", caption: "Figure 6: PanOrgan 3 Uncalibrated vs Temperature-Scaled Calibration" },
      { src: "assets/v3_conformal_coverage.png", caption: "Figure 7: PanOrgan 3 MC2 Conformal Coverage & Abstention" },
      { src: "assets/v3_mnar_stress.png", caption: "Figure 8: PanOrgan 3 MNAR Missingness Robustness Curve" }
    ],
    svgBars: null,
    verificationRows: [
      ["Zero-Leakage Split Audit", "Patient-level split verification across 8 cohorts", "Passed (0 leaked patients)"],
      ["Held-Out Test Evaluation", "Independent test split (1,425 cls / 1,246 seg)", "Verified (AUROC 0.9715, Dice 0.8355)"],
      ["Calibrated Model Weights", "Final Stage 2B checkpoint", "240 MB (Stage 2A: 715 MB)"],
      ["Conformal Coverage (alpha=0.10)", "MC2 Marginal Coverage", "93.33% (abstention 20.98%)"]
    ],
    robustnessNote: "MC2 Conformal Coverage: 93.33% marginal coverage, average set size 1.5277, abstention rate 20.98%. MNAR Stress Curve (Overall Dice): 0.8355 (0%) -> 0.8380 (20%) -> 0.7908 (40%).",
    limitations: [
      ["Missing Supervised CT & MG Labels", "Solved in PanOrgan 3.1 by integrating official TCGA-KIRC (1,200 CT) and CBIS-DDSM (88 MG) labels."],
      ["No MixUp Regularization or Explicit val_loss Logging", "Solved in PanOrgan 3.1 with MixUp (alpha=0.2, p=0.25) and per-epoch val_loss tracking."],
      ["Ultrasound Nerve Segmentation Ceiling (Dice 0.6175)", "Acoustic speckle and small brachial plexus boundaries remain challenging compared to CXR/MRI."],
      ["Unlabeled Prostate MRI (1,198 samples)", "Kept strictly unlabeled (SSL only) because official findings require restricted TCIA challenge credentials."]
    ]
  },

  "3.1": {
    id: "3.1",
    title: "PanOrgan 3.1",
    subtitle: "PanOrgan 3 Cohort — 5-Modality + Official Labels + MixUp Release",
    stage: "Version 3.1 (5-Modality Final)",
    scope: "All 5 Modalities: MRI, CT, CXR, US, MG (n = 1,629 Cls / 1,246 Seg Test)",
    status: "Completed & Evaluated",
    overview: [
      "PanOrgan 3.1 (PanOrgan 3 Cohort, Version 3.1) resolves the primary gaps of Version 3 by integrating verified official supervised labels for CT (tcga_kirc_ct via NCI GDC API) and MG (cbis_ddsm via official pathology CSVs), adding MixUp augmentation (alpha=0.2, p=0.25) on classification-only batches, and logging explicit per-epoch val_loss.",
      "Warm-started from Release 3 weights, Version 3.1 completed 5 epochs on Apple M4 (MPS) and raised overall segmentation Dice from 0.8355 to 0.8465 while expanding supervised classification evaluation to 1,629 test samples across all 5 modalities."
    ],
    mermaid: `flowchart LR
      A["<b>Release 3 Weights</b><br/>240 MB Calibrated"]:::input --> D["<b>PanOrgan 3 Cohort (3.1)</b><br/>5 Modalities: MRI, CT, CXR, US, MG"]:::core
      B["<b>+ Official CT Labels</b><br/>TCGA-KIRC (1,200)"]:::audit --> D
      C["<b>+ Official MG Labels</b><br/>CBIS-DDSM (88)"]:::audit --> D
      D --> E["<b>MixUp + val_loss</b><br/>5 Epochs (val_dice 0.8404)"]:::train
      E --> F["<b>Held-Out Test</b><br/>Seg Dice 0.8465 | AUROC 0.9593"]:::out
      classDef input fill:#eff6ff,stroke:#2563eb,color:#1e3a8a,stroke-width:1.5px;
      classDef audit fill:#ecfdf5,stroke:#059669,color:#064e3b,stroke-width:1.5px;
      classDef core fill:#eef2ff,stroke:#4f46e5,color:#312e81,stroke-width:1.5px;
      classDef train fill:#fefce8,stroke:#ca8a04,color:#713f12,stroke-width:1.5px;
      classDef out fill:#faf5ff,stroke:#9333ea,color:#581c87,stroke-width:1.5px;`,
    architectureRows: [
      ["Warm-Start Checkpoint", "Initialized from Release 3 calibrated weights (240 MB).", "Completed 5 epochs; best composite 0.8671 at Epoch 5."],
      ["MixUp Regularization", "Applied on classification-only batches (alpha = 0.2, p = 0.25).", "Reduced BUSI conformal abstention from 30.28% to 11.01%."],
      ["Validation Loss Tracking", "Logs explicit val_loss per epoch alongside val_f1, val_acc, and val_dice.", "Completed across Epochs 1–5 (train_loss 2.6559 -> 2.4089)."]
    ],
    datasetsNote: "Updated manifest holds 11,735 total samples (8,078 train / 1,839 val / 1,818 test) across 8 cohorts and 3,777 unique patients (zero-leakage verified). Held-out test evaluation covered n = 1,629 classification samples and n = 1,246 segmentation samples.",
    datasetRows: [
      ["tcga_lgg", "MRI", "3,929 (110 pts)", "Brain tumor joint seg & cls"],
      ["pulmonology_cxr", "CXR", "2,000 (2,000 pts)", "Lung seg & pulmonary cls"],
      ["ultrasound_nerve", "US", "1,500 (47 pts)", "Brachial plexus nerve seg & presence"],
      ["tcga_kirc_ct", "CT", "1,200 (15 pts)", "Renal grade G1/G2 vs G3/G4 (NCI GDC API)"],
      ["prostatex_mri", "MRI", "1,198 (25 pts)", "Unlabeled SSL / anatomy pretraining"],
      ["nih_cxr14", "CXR", "1,167 (882 pts)", "Multi-label thoracic pathology"],
      ["busi", "US", "653 (653 pts)", "Breast ultrasound joint seg & cls"],
      ["cbis_ddsm", "MG", "88 (45 pts)", "Mammography BENIGN vs MALIGNANT"]
    ],
    trainingNote: "Completed 5 epochs on Apple M4 GPU (MPS): Ep 5 reached train_loss = 2.4089, val_loss = 3.8846, val_dice = 0.8404, val_f1 = 0.8938, val_acc = 87.15% (best composite = 0.8671).",
    epochRows: [
      ["1", "2.6559", "3.7232", "0.8089", "0.8707", "84.90%"],
      ["2", "2.6242", "3.8593", "0.8339", "0.8643", "84.90%"],
      ["3", "2.4957", "4.3587", "0.8213", "0.8219", "81.91%"],
      ["4", "2.4389", "3.9015", "0.8338", "0.8832", "86.48%"],
      ["5 (Best)", "2.4089", "3.8846", "0.8404", "0.8938", "87.15%"]
    ],
    conformalRows: [
      ["nih_cxr14", "179", "100.00%", "1.00", "0.00%"],
      ["tcga_lgg", "537", "94.97%", "1.00", "0.00%"],
      ["busi", "109", "91.74%", "1.13", "11.01%"],
      ["pulmonology_cxr", "334", "87.72%", "1.00", "0.00%"],
      ["cbis_ddsm", "15", "93.33%", "3.93", "100.00%"],
      ["tcga_kirc_ct", "189", "85.19%", "3.41", "100.00%"],
      ["ultrasound_nerve", "266", "93.98%", "3.60", "100.00%"],
      ["Overall", "1,629", "92.51%", "1.74", "29.59%"]
    ],
    classificationRows: [
      ["AUROC", "0.9590", "0.9593"],
      ["Accuracy", "85.57%", "85.57%"],
      ["Macro-F1", "0.8814", "0.8814"],
      ["Macro Precision", "0.8832", "0.8832"],
      ["Macro Sensitivity (Recall)", "0.8802", "0.8802"],
      ["Macro Specificity", "0.9358", "0.9358"],
      ["ECE", "0.0611", "0.0784"],
      ["Brier Score", "0.2483", "0.2499"]
    ],
    segmentationRows: [
      ["pulmonology_cxr", "334", "0.9664 (0.9631–0.9695)", "0.9366", "8.24"],
      ["tcga_lgg", "537", "0.8901 (0.8714–0.9087)", "0.8560", "6.13"],
      ["busi", "109", "0.7677 (0.7165–0.8155)", "0.6871", "37.30"],
      ["ultrasound_nerve", "266", "0.6402 (0.5875–0.6897)", "0.6028", "10.91"],
      ["Overall", "1,246", "0.8465 (0.8294–0.8627)", "0.8088", "10.30"]
    ],
    images: [
      { src: "assets/v31_confusion_matrix.png", caption: "Figure 1: PanOrgan 3.1 Test Confusion Matrix (n = 1,629)" },
      { src: "assets/v31_train_val_loss.png", caption: "Figure 2: PanOrgan 3.1 Training & Validation Loss (5 Epochs)" },
      { src: "assets/v31_val_accuracy_f1.png", caption: "Figure 3: PanOrgan 3.1 Validation Accuracy, Macro-F1 & Dice (5 Epochs)" },
      { src: "assets/v31_precision_recall_metrics.png", caption: "Figure 4: PanOrgan 3.1 Test Accuracy, Precision, Recall & F1 (n = 1,629)" },
      { src: "assets/v31_segmentation_dice.png", caption: "Figure 5: PanOrgan 3.1 Segmentation Dice & IoU (n = 1,246)" },
      { src: "assets/v31_conformal_coverage.png", caption: "Figure 6: PanOrgan 3.1 MC2 Conformal Coverage & Abstention (7 Cohorts)" },
      { src: "assets/v31_cohort_distribution.png", caption: "Figure 7: PanOrgan 3.1 Sharded Cohort Distribution (11,735 Samples)" },
      { src: "assets/v31_modality_breakdown.png", caption: "Figure 8: PanOrgan 3.1 Five-Modality Coverage (MRI, CXR, US, CT, MG)" },
      { src: "assets/v31_split_audit.png", caption: "Figure 9: PanOrgan 3.1 Zero-Leakage Train / Val / Test Splits" },
      { src: "assets/v31_patient_diversity.png", caption: "Figure 10: PanOrgan 3.1 Unique Patient Count per Cohort (3,777 Patients)" }
    ],
    svgBars: null,
    verificationRows: [
      ["Zero-Leakage Split Audit", "Patient-level split verification across 8 cohorts", "Passed (0 leaked patients)"],
      ["Official CT & MG Sources", "NCI GDC API (537 pts) & CBIS-DDSM Pathology CSVs", "Verified (1,200 CT + 88 MG shards)"],
      ["Held-Out Test Evaluation", "Independent 5-modality test split (1,629 cls / 1,246 seg)", "Completed (Overall Dice 0.8465, AUROC 0.9593)"],
      ["Conformal Coverage (alpha=0.10)", "MC2 Marginal Coverage across 7 supervised cohorts", "92.51% (BUSI abstention dropped to 11.01%)"]
    ],
    robustnessNote: "MC2 Conformal Coverage: 92.51% marginal coverage across all 7 supervised cohorts (avg set size 1.7397, abstention 29.59%). Overall segmentation Dice improved from 0.8355 (V3) to 0.8465 (V3.1).",
    limitations: [
      ["Small Mammography Sample Pool (88 samples / 45 pts)", "Solved in PanOrgan 3.2 by adding 2,608 CMMD mammography scans (1,430 patients)."],
      ["Low Unique Patient Count in tcga_kirc_ct (15 pts / 1,200 slices)", "Solved in PanOrgan 3.2 by adding 2,487 Kidney CT Abnormality slices across 829 patients."],
      ["Temperature Scaling ECE Slight Rise (0.0611 -> 0.0784)", "Solved in PanOrgan 3.2 (ECE dropped to 0.0396 at Ep 3 and 0.0479 post-calibration at Ep 4)."],
      ["Unlabeled prostatex_mri (1,198 samples)", "Excluded from supervised loss due to restricted TCIA challenge credentials."]
    ]
  },

  "3.2": {
    id: "3.2",
    title: "PanOrgan 3.2",
    subtitle: "PanOrgan 3 Cohort — Balanced 20,175-Shard 5-Modality Release",
    stage: "Version 3.2 (Balanced 5-Modality Release)",
    scope: "All 5 Modalities: MRI, CT, CXR, US, MG (20,175 Shards / 9,381 Patients)",
    status: "Completed & Evaluated (Ep 3 & Ep 4 Best Checkpoints Verified)",
    overview: [
      "PanOrgan 3.2 (PanOrgan 3 Cohort, Version 3.2) resolves the class imbalance and small-cohort bottlenecks of Version 3.1 without removing any existing data. It adds 2,608 CMMD mammograms (1,430 patients), 2,487 Kidney CT Abnormality slices (829 patients), and 3,345 unused COVID-19 Radiography Class 2/3 CXR scans (3,345 patients), expanding the zero-leakage manifest to 20,175 samples across 9,381 unique patients.",
      "Training warm-started from PanOrgan 3.1 best.pt with lr=3e-5 Cosine Annealing, MixUp (alpha=0.2, p=0.25), Class-Weighted Loss, and Early Stopping (patience=2) across 6 epochs, achieving peak composite score 0.8737 at Epoch 4 and overall held-out test segmentation Dice 0.8824 (Ep 3) / 0.8800 (Ep 4)."
    ],
    mermaid: `flowchart LR
      A["<b>PanOrgan 3.1</b><br/>11,735 Shards (best.pt)"]:::input --> D["<b>PanOrgan 3.2 Balanced</b><br/>20,175 Shards | 9,381 Patients"]:::core
      B["<b>+ CMMD (MG)</b><br/>2,608 Scans (1,430 pts)"]:::audit --> D
      C["<b>+ Kidney CT & CXR Ext</b><br/>2,487 CT + 3,345 CXR"]:::audit --> D
      D --> E["<b>lr=3e-5 + MixUp + ES</b><br/>12,895 Train / 3,044 Val / 3,227 Test"]:::train
      E --> F["<b>Completed (6 Epochs)</b><br/>Seg Dice 0.8824 | AUROC 0.9485"]:::out
      classDef input fill:#eff6ff,stroke:#2563eb,color:#1e3a8a,stroke-width:1.5px;
      classDef audit fill:#ecfdf5,stroke:#059669,color:#064e3b,stroke-width:1.5px;
      classDef core fill:#eef2ff,stroke:#4f46e5,color:#312e81,stroke-width:1.5px;
      classDef train fill:#fefce8,stroke:#ca8a04,color:#713f12,stroke-width:1.5px;
      classDef out fill:#faf5ff,stroke:#9333ea,color:#581c87,stroke-width:1.5px;`,
    architectureRows: [
      ["Warm-Start & Gentle LR", "Initialized from PanOrgan 3.1 best.pt with lr = 3.0e-5 Cosine Decay + Weight Decay 0.05.", "Completed 6 epochs; val_loss reached minimum 2.9562 at Ep 3 and best composite 0.8737 at Ep 4."],
      ["Balanced Sampling & Loss", "BalancedModalityBatchSampler (alpha=0.35) + Class-Weighted Cross-Entropy + MixUp (0.2, p=0.25).", "Class 2 boosted 690 -> 3,190; Class 3 boosted 500 -> 1,345."],
      ["Early Stopping & Dual Gate", "Early Stopping (patience = 2) + full held-out test verification at Ep 3 and Ep 6 (evaluating Ep 4 best.pt).", "Both Ep 3 (Dice 0.8824, ECE 0.0396) and Ep 4 (Dice 0.8800, ECE 0.0479) verified."]
    ],
    datasetsNote: "Expanded zero-leakage manifest holds 20,175 samples (12,895 sup train / 3,044 sup val / 3,227 test) across 11 cohorts and 9,381 unique patients (leakage_count = 0).",
    datasetRows: [
      ["tcga_lgg", "MRI", "3,929 (110 pts)", "Brain tumor joint seg & cls"],
      ["pulmonology_cxr_ext", "CXR", "3,345 (3,345 pts)", "Class 2 (2,500) & Class 3 (845) booster"],
      ["cmmd_mg", "MG", "2,608 (1,430 pts)", "Mammography Benign vs Malignant"],
      ["kidney_ct_abn", "CT", "2,487 (829 pts)", "Normal vs Abnormal Renal CT"],
      ["pulmonology_cxr", "CXR", "2,000 (2,000 pts)", "Lung seg & 4-class pulmonary cls"],
      ["ultrasound_nerve", "US", "1,500 (47 pts)", "Brachial plexus nerve seg & presence"],
      ["tcga_kirc_ct", "CT", "1,200 (15 pts)", "Renal grade G1/G2 vs G3/G4"],
      ["prostatex_mri", "MRI", "1,198 (25 pts)", "Unlabeled SSL pretraining"],
      ["nih_cxr14", "CXR", "1,167 (882 pts)", "Multi-label thoracic pathology"],
      ["busi", "US", "653 (653 pts)", "Breast ultrasound joint seg & cls"],
      ["cbis_ddsm", "MG", "88 (45 pts)", "Mammography pathology"]
    ],
    trainingNote: "Completed all 6 epochs on Apple M4 GPU (MPS) with Early Stopping retaining the Epoch 4 best checkpoint (composite = 0.8737). Both the Epoch 3 Pre-Deployment Validation (AUROC 0.9485, Macro-F1 0.8574, ECE 0.0396, Dice 0.8824) and the Final Epoch 4 Best Checkpoint Evaluation (AUROC 0.9431, Macro-F1 0.8479, Calibrated ECE 0.0479, Dice 0.8800, HD95 9.52 mm) were verified on the held-out test split (n = 3,038 cls / 1,804 seg).",
    epochRows: [
      ["1", "2.4145", "3.0026", "0.8781", "0.8396", "78.29%"],
      ["2", "2.3504", "2.9890", "0.8632", "0.8545", "80.19%"],
      ["3 (Validated)", "2.2922", "2.9562", "0.8758", "0.8648", "81.31%"],
      ["4 (Best Composite)", "2.2451", "3.1748", "0.8814", "0.8660", "81.50%"],
      ["5", "2.1670", "3.2087", "0.8783", "0.8675", "81.64%"],
      ["6 (Early Stop)", "2.1153", "3.2206", "0.8807", "0.8659", "81.50%"]
    ],
    classificationHeaders: ["Metric", "Ep 3 Validated (T = 1.0238)", "Ep 4 Best Calibrated (T = 1.1851)"],
    classificationRows: [
      ["AUROC", "0.9485", "0.9431"],
      ["Accuracy", "80.15%", "78.97%"],
      ["Macro-F1", "0.8574", "0.8479"],
      ["Macro Precision", "0.8634", "0.8466"],
      ["Macro Sensitivity (Recall)", "0.8533", "0.8503"],
      ["Macro Specificity", "0.9210", "0.9176"],
      ["ECE (Calibrated)", "0.0396", "0.0479"],
      ["Brier Score (Calibrated)", "0.2848", "0.3253"]
    ],
    segmentationRows: [
      ["pulmonology_cxr", "334", "0.9707 (0.9680–0.9731)", "0.9440", "7.34"],
      ["pulmonology_cxr_ext", "558", "0.9611 (0.9585–0.9636)", "0.9267", "9.67"],
      ["tcga_lgg", "537", "0.8944 (0.8742–0.9121)", "0.8615", "5.95"],
      ["busi", "109", "0.7691 (0.7188–0.8194)", "0.6900", "35.00"],
      ["ultrasound_nerve", "266", "0.6125 (0.5574–0.6640)", "0.5794", "8.90"],
      ["Overall (Ep 4 / Ep 3: 0.8824)", "1,804", "0.8800 (0.8683–0.8913)", "0.8450", "9.52"]
    ],
    conformalRows: [
      ["nih_cxr14", "179", "100.00%", "1.00", "0.00%"],
      ["tcga_lgg", "537", "94.97%", "1.00", "0.00%"],
      ["pulmonology_cxr_ext", "558", "94.44%", "1.00", "0.00%"],
      ["pulmonology_cxr", "334", "91.02%", "1.00", "0.00%"],
      ["busi", "109", "90.83%", "1.94", "77.06%"],
      ["cmmd_mg", "437", "89.93%", "1.64", "64.07%"],
      ["kidney_ct_abn", "414", "92.27%", "1.72", "72.22%"],
      ["cbis_ddsm", "15", "100.00%", "3.47", "100.00%"],
      ["tcga_kirc_ct", "189", "77.78%", "3.33", "100.00%"],
      ["ultrasound_nerve", "266", "86.47%", "2.48", "100.00%"],
      ["Overall (Ep 3 / Ep 4: 90.22%)", "3,038", "91.71%", "1.51", "37.29%"]
    ],
    images: [
      { src: "assets/v32_train_progress.png", caption: "Figure 1: PanOrgan 3.2 Training & Validation Progress (6 Epochs)" },
      { src: "assets/v32_confusion_matrix.png", caption: "Figure 2: PanOrgan 3.2 Held-Out Test Confusion Matrix (n = 3,038)" },
      { src: "assets/v32_precision_recall_metrics.png", caption: "Figure 3: PanOrgan 3.2 Test Classification Metrics (n = 3,038)" },
      { src: "assets/v32_segmentation_dice.png", caption: "Figure 4: PanOrgan 3.2 Segmentation Dice & IoU (n = 1,804)" },
      { src: "assets/v32_conformal_coverage.png", caption: "Figure 5: PanOrgan 3.2 MC2 Conformal Coverage (10 Cohorts)" },
      { src: "assets/v32_mnar_stress.png", caption: "Figure 6: PanOrgan 3.2 MNAR Missingness Curve (0% -> 20% -> 40%)" },
      { src: "assets/v32_cohort_distribution.png", caption: "Figure 7: PanOrgan 3.2 Cohort Distribution (20,175 Samples / 9,381 Patients)" },
      { src: "assets/v32_class_balance.png", caption: "Figure 8: Class Distribution Expansion (PanOrgan 3.1 vs 3.2)" },
      { src: "assets/v32_modality_expansion.png", caption: "Figure 9: 5-Modality Sample Expansion (PanOrgan 3.1 vs 3.2)" }
    ],
    svgBars: null,
    verificationRows: [
      ["Zero-Leakage Split Audit", "Patient-level split verification across 11 cohorts (9,381 pts)", "Passed (0 leaked patients)"],
      ["6-Epoch Balanced Training", "train_v32_balanced.py on Apple M4 MPS", "Completed (Best Composite 0.8737 at Ep 4)"],
      ["Held-Out Test Evaluation", "v32_ep3_predeployment_validation.json & v32_complete_thesis_report.json", "Verified (Dice 0.8824 / 0.8800, AUROC 0.9485 / 0.9431)"]
    ],
    robustnessNote: "MC2 Conformal Coverage: 91.71% (Ep 3) / 90.22% (Ep 4) marginal coverage across 10 supervised cohorts (n = 3,038). MNAR Stress Curve (Overall Dice, n = 1,804): 0.8824 (0%) -> 0.8806 (20%) -> 0.8400 (40%).",
    limitations: [
      ["Shared 4-Class Output Head Across Binary & Multi-Class Cohorts", "Addressed in PanOrgan 4 via dedicated task-group classification heads (PanOrgan4MultiHead) and cohort-active class masking."],
      ["Class 0/1 Cross-Modality Confusion in Newly Added Binary CT/MG Cohorts", "Addressed via per-modality temperature scaling and task-specific linear heads."],
      ["Unlabeled prostatex_mri (1,198 samples)", "Excluded from supervised loss due to restricted TCIA challenge credentials."]
    ]
  },
  "4": {
    id: "4",
    title: "PanOrgan 4",
    subtitle: "Multi-Head Task-Isolated Foundation Model (100% Supervised 11-Cohort)",
    stage: "Version 4 (Active Training)",
    scope: "5 Task-Group Heads & 14-Label Supervision",
    status: "In Progress (Training Live)",
    overview: [
      "PanOrgan 4 resolves the shared 4-class classification bottleneck observed in PanOrgan 3.2 by introducing PanOrgan4MultiHead: a unified PanOrganNetV2 backbone paired with 5 dedicated task-group classification heads (cxr_4class, busi_3class, mg_binary, ct_binary, mri_us_binary) and a 14-label BCEWithLogitsLoss objective.",
      "Additionally, PanOrgan 4 resolves all previously unlabeled prostatex_mri cases (1,198 / 1,198 matched via official PROSTATEx lesion/clinical labels), bringing the 20,175-sample / 9,381-patient master manifest (master_manifest_v4.parquet) to 100% supervised coverage (0 unlabeled samples)."
    ],
    mermaid: `flowchart LR
    A["PanOrgan 3.2 Backbone<br/>Warm-Start (best.pt)"] --> B["PanOrgan4MultiHead<br/>Shared Encoder + 5 Task Heads"]
    B --> C["cxr_4class & busi_3class<br/>Multi-Class Thoracic/US"]
    B --> D["mg_binary, ct_binary, mri_us_binary<br/>Dedicated Binary Screening"]
    C & D --> E["14-Label BCE + Dice/Focal<br/>Patient-Level 0-Leakage (9,381 pts)"]`,
    architectureRows: [
      ["PanOrgan4MultiHead", "Wraps PanOrganNetV2 backbone with 5 isolated task-group heads (cxr_4class, busi_3class, mg_binary, ct_binary, mri_us_binary)", "Implemented (V4/scripts/PanOrgan_4_train.py)"],
      ["14-Label Supervision", "Replaces collapsed 4-class mapping with task-specific 14-label BCEWithLogitsLoss + per-task masking", "Implemented (master_manifest_v4.parquet)"],
      ["Isolated PanOrgan_4 SSD Storage", "Dedicated checkpoints, logs, and thesis reports under /Volumes/Shoaib/PanOrgan_4/", "Active (PID 40507, caffeinate + 25-step resume)"]
    ],
    datasetsNote: "All 20,175 samples across 9,381 unique patients (Train: 13,707 | Val: 3,241 | Test: 3,227) are 100% supervised (0 unlabeled) with strict patient-level zero-leakage splitting.",
    datasetHeaders: ["Cohort / Split", "Modality", "Samples", "Clinical Task / Label Mapping"],
    datasetRows: [
      ["prostatex_mri (Newly Labeled)", "MRI (Prostate)", "1,198", "1,198 / 1,198 matched via official PROSTATEx labels (mri_us_binary)"],
      ["pulmonary_cxr + cxr_multiclass", "CXR (Thoracic)", "6,796", "4-class thoracic screening (cxr_4class)"],
      ["busi_breast_us + ultrasound_nerve", "Ultrasound", "6,389", "3-class breast lesion (busi_3class) + nerve seg/cls (mri_us_binary)"],
      ["cbis_ddsm + vindr_mammo_mg + cmmd_mg", "Mammography (MG)", "1,852", "Binary breast malignancy screening (mg_binary)"],
      ["kits23_ct + tcga_kirc_ct + kidney_ct_mri", "CT / Renal", "3,940", "Renal lesion segmentation & binary classification (ct_binary)"],
      ["Total Supervised Manifest (V4)", "5 Modalities (11 Cohorts)", "20,175 (9,381 pts)", "Train: 13,707 | Val: 3,241 | Test: 3,227 | Unlabeled: 0"]
    ],
    trainingNote: "PanOrgan 4 is warm-started from /Volumes/Shoaib/PanOrgan_V3/checkpoints/v32_balanced/best.pt and is currently training live for 6 epochs (Early Stopping patience = 2, 25-step crash-resume checkpointing) on Apple M4 MPS. Final test metrics, MNAR 0/20/40% curves, MC2 conformal coverage, and diagnostic figures will be populated automatically upon completion.",
    epochRows: null,
    classificationRows: null,
    segmentationRows: null,
    conformalRows: null,
    images: [],
    svgBars: null,
    verificationRows: [
      ["V4 Label Reconciliation", "V4/scripts/PanOrgan_4_labels.py -> master_manifest_v4.parquet", "Verified (20,175 samples, 9,381 pts, 0 unlabeled)"],
      ["PROSTATEx Official Label Match", "1,198 / 1,198 prostatex_mri samples matched", "Verified (100% matched)"],
      ["Live Training Execution", "/Volumes/Shoaib/PanOrgan_4/experiments/panorgan_4_pipeline.log", "In Progress (PID 40507, Epoch 1/6, 25-step resume)"]
    ],
    robustnessNote: "PanOrgan 4 training is currently in progress. Upon completion, final held-out test metrics (Dice, AUROC, Macro-F1, ECE, MC2 Conformal Coverage, and MNAR 0/20/40% stress evaluation) will be generated in /Volumes/Shoaib/PanOrgan_4/experiments/thesis_reports/panorgan_4_complete_thesis_report.json.",
    limitations: [
      ["Training In Progress (Epoch 1/6)", "Final quantitative metrics, confusion matrices, and diagnostic plots will be added once PanOrgan_4_train.py finishes."],
      ["Multi-Head Calibration Pending", "Per-task temperature scaling and MC2 conformal calibration run automatically after the final training epoch."]
    ]
  }
};
