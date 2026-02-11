# Deforestation Detection (CW2) — Baseline Replication + Contextual Adaptation

This repository is my **AI for Sustainable Development — Coursework 2 (50%)** submission.

It includes:

- **Part A-1 (Replication):** reproduce the **paper baseline** U‑Net and Attention U‑Net on the **Amazon 4‑band** dataset and match the paper’s reported metrics within **±5%**.
- **Part A-2~A-5 (Local context):** adapt the same segmentation approach to **East Kalimantan (Indonesia)** for **2023 deforestation**, including **dataset curation**, **model/training adaptations**, **statistical significance testing**, and **failure‑case analysis**.

---

## Results summary (what to look at first)

### A) Baseline replication (Amazon 4‑band, validation set)

| Model | Source | Accuracy | Precision | Recall | F1 / Dice | IoU |
|---|---|---:|---:|---:|---:|---:|
| U‑Net (4‑band) | Paper (reported) | 0.9395 | 0.9766 | 0.9395 | 0.9441 | — |
| U‑Net (4‑band) | Our replication | 0.9615 | 0.9758 | 0.9497 | 0.9626 | 0.9278 |
| Attention U‑Net (4‑band) | Paper (reported) | 0.9448 | 0.9790 | 0.9448 | 0.9459 | — |
| Attention U‑Net (4‑band) | Our replication | 0.9611 | 0.9807 | 0.9441 | 0.9621 | 0.9269 |

✅ All reported paper metrics are reproduced **within ±5%** (see `baseline_replication/reproduce_baseline.ipynb`, Section 10).

---

### B) Local context (East Kalimantan 2023, **test set**, threshold tuned on validation)

**Test set size:** 352 patches, of which **247** are GT‑positive patches (contain ≥1 deforestation pixel).  
**Split sizes (train/val/test):** 1159/317/352 patches.

**U‑Net**

| Setting | Threshold (val‑tuned) | Dice (GT‑positive patches) | IoU (GT‑positive patches) | Dice_global | IoU_global |
|---|---:|---:|---:|---:|---:|
| Baseline (direct transfer) | 0.15 | 0.1298 (CI95 0.1172–0.1433) | 0.0730 | 0.1261 | 0.0673 |
| Adapted (imbalance + tuning) | 0.40 | 0.1764 (CI95 0.1574–0.1953) | 0.1061 | 0.1630 | 0.0887 |

**Attention U‑Net**

| Setting | Threshold (val‑tuned) | Dice (GT‑positive patches) | IoU (GT‑positive patches) | Dice_global | IoU_global |
|---|---:|---:|---:|---:|---:|
| Baseline (direct transfer) | 0.15 | 0.1235 (CI95 0.1120–0.1357) | 0.0687 | 0.1300 | 0.0695 |
| Adapted (imbalance + tuning) | 0.55 | 0.1708 (CI95 0.1526–0.1887) | 0.1023 | 0.1563 | 0.0848 |

---

### C) Statistical significance (local context)

All tests are **paired per‑patch** comparisons on the **same test patches**.

**U‑Net (adapted − baseline)** from `local_context/contextual_adaptation_kalimantan_2023_en_updated_downloads.ipynb`:

- **IoU (all test patches, N=352)**: mean diff = 0.0232, p(t‑test)=4.65e-10, p(Wilcoxon)=8.06e-28
- **IoU (GT‑positive only, N=247)**: mean diff = 0.0330, p(t‑test)=2.92e-10, p(Wilcoxon)=2.58e-27
- **Dice (all test patches, N=352)**: mean diff = 0.0327, p(t‑test)=1.96e-12, p(Wilcoxon)=5.24e-28
- **Dice (GT‑positive only, N=247)**: mean diff = 0.0466, p(t‑test)=8.93e-13, p(Wilcoxon)=1.27e-27

**Attention U‑Net (adapted − baseline, IoU per patch)**: mean diff = 0.0236, p(t‑test)=8.09e-09, p(Wilcoxon)=5.23e-26.

---

## Repository structure

```
.
├── baseline_replication/
│   ├── reproduce_baseline.ipynb
│   └── attention_unet/                   # reference implementation + scripts
├── local_context/
│   ├── contextual_adaptation_kalimantan_2023_en_updated_downloads.ipynb   # (UPDATED)
│   ├── contextual_adaptation_kalimantan_2023_en_updated_downloads_(1) (2).ipynb  # older copy
│   ├── unet_baseline_best.pt
│   └── unet_adapted_best.pt
├── requirements.txt
└── README.md
```

---

## Part A‑1 — Baseline replication (Amazon 4‑band)

### Notebook
Run: `baseline_replication/reproduce_baseline.ipynb`

This notebook:
- loads preprocessed Amazon `.npy` patches (`512×512×4`)
- trains **U‑Net** (30 epochs) and **Attention U‑Net** (60 epochs)
- evaluates on validation with Accuracy/Precision/Recall/F1/IoU
- compares against paper metrics with a **±5% tolerance check**

### Data (Zenodo)
The reference preprocessing expects the Zenodo **Amazon 4‑band** dataset to be placed under:

`baseline_replication/attention_unet/AMAZON/`

Then run preprocessing:

```bash
cd baseline_replication/attention_unet
python preprocess-4band-amazon-data.py
# outputs: amazon-processed-large/{training,validation,test}/...
```

---

## Part A‑2 / A‑3 — Local context data: East Kalimantan (Indonesia), 2023

### SDG alignment
- **SDG 13 (Climate Action)** and **SDG 15 (Life on Land)**

### Data sources (open/public)
- **Sentinel‑2 L2A Surface Reflectance (harmonized)**, 2023 median composite (4 bands)
- **Hansen Global Forest Change (GFC)**: `lossyear == 2023` (supervision mask, 30m)

### Files expected by the notebook
Place exports under: `data_raw/`

Required:
- `HANSEN_LOSS_2023_UTM_30m.tif`

Sentinel‑2 can be provided in **either** form:
- a single mosaic: `S2_EKAL_2023_MEDIAN_4B_UTM.tif`, **or**
- multiple tiles: `S2_EKAL_2023_MEDIAN_4B_UTM-*.tif` (the notebook mosaics them)

Optional:
- `HANSEN_TREECOVER2000_UTM_30m.tif` (if you want to mask by forest cover threshold)

> The notebook includes both a **GEE export helper** (preferred) and a **direct download** option for Hansen tiles.

---

## Part A‑4 — What “baseline” vs “adapted” means (local context)

Both are trained/evaluated on the same Kalimantan 2023 patch dataset.

- **Baseline (direct transfer):**
  - model: U‑Net / Attention U‑Net
  - loss: plain `BCEWithLogitsLoss`
  - evaluation at default threshold (0.5) can be degenerate under imbalance; therefore we tune the threshold on validation.

- **Adapted (contextual adaptation):**
  - loss: **weighted** `BCEWithLogitsLoss(pos_weight)` + **Soft Dice** (handles extreme class imbalance)
  - threshold tuning on validation (maximize Dice on GT‑positive patches)
  - same patch pipeline + reproducible seeds

See Section “Loss functions” and “Threshold tuning” inside:
`local_context/contextual_adaptation_kalimantan_2023_en_updated_downloads.ipynb`.

---

## Part A‑5 — Evaluation + failure cases

Implemented in the local context notebook:

- Per‑patch metrics (IoU/Dice/Precision/Recall)
- Bootstrap **95% CI** on Dice over GT‑positive patches
- Paired **t‑test** and **Wilcoxon signed‑rank** test on per‑patch differences
- Failure case visualization (input band / GT / prediction / error map)

---

## Ethics, limitations, sustainability (local context)

- **Label mismatch:** Hansen loss is **30m**, Sentinel‑2 is **10m** → mixed pixels & boundary noise.
- **Temporal ambiguity:** Hansen `lossyear=2023` indicates year of loss, not exact date; cloud gaps can affect composites.
- **Class imbalance:** deforestation pixels are rare → requires weighted losses and careful thresholding.
- **Responsible use:** outputs are decision support; local validation and community safeguards are needed before real deployment.

---

## Part B — Poster & presentation

Prepare a digital academic poster (template provided by the course) and include a link to this repository.
Presentation guideline: ~30s paper + ~30s new context + ~1min results/interpretation, then Q&A.

---

## Reproduce everything (checklist)

### Baseline replication (Amazon)
- [ ] Place Zenodo dataset under `baseline_replication/attention_unet/AMAZON/`
- [ ] Run `python preprocess-4band-amazon-data.py`
- [ ] Run `baseline_replication/reproduce_baseline.ipynb` end‑to‑end

### Local context (Kalimantan 2023)
- [ ] Put GeoTIFFs under `data_raw/` (see filenames above)
- [ ] Run `local_context/contextual_adaptation_kalimantan_2023_en_updated_downloads.ipynb` end‑to‑end
