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
    datasetRows: null,
    trainingNote: "PanOrgan 1 served as the architectural specification and initial baseline prototype phase. Unified multi-cohort joint training and conformal calibration began in PanOrgan 2.",
    classificationRows: null,
    segmentationRows: null,
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
    trainingNote: "Trained on Apple M4 GPU (MPS) with external SSD sharding. Post-hoc temperature scaling optimized T = 1.6201 on validation logits, followed by MC2 conformal calibration targeting 90% marginal coverage (alpha = 0.10).",
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
    subtitle: "PanOrgan 3 Cohort — 5-Modality + Official Labels + MixUp Upgrade",
    stage: "Version 3.1 (Active Fine-Tune)",
    scope: "All 5 Modalities: MRI, CT, CXR, US, MG (11,735 Shards)",
    status: "In Progress (PID 5831 on M4 MPS)",
    overview: [
      "PanOrgan 3.1 (PanOrgan 3 Cohort, Version 3.1) directly resolves the primary gaps of Version 3 by integrating verified official supervised labels for CT (tcga_kirc_ct via NCI GDC API) and MG (cbis_ddsm via official pathology CSVs), adding MixUp augmentation (alpha=0.2, p=0.25) on classification-only batches, and logging explicit per-epoch val_loss.",
      "Rather than restarting from scratch, Version 3.1 warm-starts from the calibrated Release 3 weights."
    ],
    mermaid: `flowchart LR
      A["<b>Release 3 Weights</b><br/>240 MB Calibrated"]:::input --> D["<b>PanOrgan 3 Cohort (3.1)</b><br/>5 Modalities: MRI, CT, CXR, US, MG"]:::core
      B["<b>+ Official CT Labels</b><br/>TCGA-KIRC (1,200)"]:::audit --> D
      C["<b>+ Official MG Labels</b><br/>CBIS-DDSM (88)"]:::audit --> D
      D --> E["<b>MixUp + val_loss</b><br/>7,266 Train / 1,642 Val / 1,818 Test"]:::train
      E --> F["<b>Active Run (PID 5831)</b><br/>M4 MPS (5 Epochs)"]:::out
      classDef input fill:#eff6ff,stroke:#2563eb,color:#1e3a8a,stroke-width:1.5px;
      classDef audit fill:#ecfdf5,stroke:#059669,color:#064e3b,stroke-width:1.5px;
      classDef core fill:#eef2ff,stroke:#4f46e5,color:#312e81,stroke-width:1.5px;
      classDef train fill:#fefce8,stroke:#ca8a04,color:#713f12,stroke-width:1.5px;
      classDef out fill:#faf5ff,stroke:#9333ea,color:#581c87,stroke-width:1.5px;`,
    architectureRows: [
      ["Warm-Start Checkpoint", "Initialized from Release 3 calibrated weights (240 MB).", "Active run PID 5831 (5 epochs, checkpoint every 20 steps)."],
      ["MixUp Regularization", "Applied on classification-only batches (alpha = 0.2, p = 0.25).", "Active during 3.1 fine-tuning."],
      ["Validation Loss Tracking", "Logs explicit val_loss per epoch alongside val_f1, val_acc, and val_dice.", "Streaming to live pipeline log."]
    ],
    datasetsNote: "Updated manifest holds 11,735 total samples (8,078 train / 1,839 val / 1,818 test) across 8 cohorts and 3,777 unique patients (zero-leakage verified). Active 3.1 supervised split: 7,266 train / 1,642 val / 1,818 test.",
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
    trainingNote: "Running on Apple M4 GPU (MPS) with caffeinate protection (PID 5831, 908 steps/epoch, 5 epochs).",
    classificationRows: null,
    segmentationRows: null,
    images: [
      { src: "assets/v31_cohort_distribution.png", caption: "Figure 1: PanOrgan 3.1 Sharded Cohort Distribution (11,735 Samples)" },
      { src: "assets/v31_modality_breakdown.png", caption: "Figure 2: PanOrgan 3.1 Five-Modality Coverage (MRI, CXR, US, CT, MG)" },
      { src: "assets/v31_split_audit.png", caption: "Figure 3: PanOrgan 3.1 Zero-Leakage Train / Val / Test Splits" },
      { src: "assets/v31_patient_diversity.png", caption: "Figure 4: PanOrgan 3.1 Unique Patient Count per Cohort (3,777 Patients)" }
    ],
    svgBars: null,
    verificationRows: [
      ["Zero-Leakage Split Audit", "Patient-level split verification across 8 cohorts", "Passed (0 leaked patients)"],
      ["Official CT Clinical Source", "NCI GDC API Clinical Grades", "537 patients (1,200 sharded CT slices)"],
      ["Official MG Clinical Source", "CBIS-DDSM Pathology CSVs", "45 patients (88 sharded MG scans)"],
      ["Live Training Process", "PID 5831 on M4 MPS", "Active (Epoch 1/5)"]
    ],
    robustnessNote: "Evaluation Status: Training is currently in progress (PID 5831, Epoch 1/5). Final 3.1 test-set AUROC, Macro-F1, Dice, ECE, MC2 conformal coverage, and MNAR stress metrics will be populated once all 5 epochs and evaluation complete.",
    limitations: [
      ["Small Mammography Sample Pool (88 samples / 45 pts)", "Target for future expansion with additional curated MG datasets."],
      ["Low Unique Patient Count in tcga_kirc_ct (15 pts / 1,200 slices)", "Slice-level volume is high, but patient diversity is limited."],
      ["Unlabeled prostatex_mri (1,198 samples)", "Excluded from supervised loss due to restricted TCIA challenge credentials."],
      ["Final 3.1 Evaluation Pending", "Awaiting completion of PID 5831 (5 epochs) for held-out test metrics."]
    ]
  }
};
