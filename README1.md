# Deforestation Detection — CW2 (Baseline Replication + Contextual Adaptation)

This repository is my **Coursework 2 (50%)** submission for *AI for Sustainable Development*.

It contains two parts:

- **Part A-1 (Baseline replication):** reproduce the published **U‑Net** and **Attention U‑Net** deforestation segmentation baselines on the **4‑band Amazon Sentinel‑2 dataset**.
- **Part A-2…A-5 (Contextual adaptation):** adapt the same approach to a **new context** — **East Kalimantan (Borneo), Indonesia** — focusing on **2023 forest loss segmentation** with **Sentinel‑2 SR + Hansen GFC**.
- **Part B (Poster + presentation):** a digital poster (template provided by the course) with a link to this GitHub repo.

> 中文一句话：这个 repo 里 baseline 复现 + Kalimantan(2023) 本地化实验都能从 notebook 直接跑通；README 的数据、阈值、超参、结果表均与 notebook **已执行输出**对齐。

---

## 0) Repository structure

```
.
├── README.md
├── requirements.txt
├── baseline_replication/
│   ├── reproduce_baseline.ipynb
│   └── attention_unet/                      # cloned / referenced baseline repo
│       ├── preprocess-4band-amazon-data.py
│       ├── models/                          # reference weights from original repo
│       └── README.md                        # original repo instructions
└── local_context/
    ├── contextual_adaptation_kalimantan_2023_en_updated_downloads_(1) (2).ipynb
    ├── unet_baseline_best.pt
    └── unet_adapted_best.pt
```

---

## 1) Baseline paper (what we replicate)

**Paper:** John, D., & Zhang, C. (2022). *An attention-based U-Net for detecting deforestation within satellite sensor imagery.*  
**Task:** binary semantic segmentation (deforestation vs non-deforestation).  
**Input:** Sentinel‑2 **4 bands** (Green, Red, NIR, SWIR).  
**Models:** U‑Net, Attention U‑Net.  
**Metrics used in the replication notebook:** Accuracy, Precision, Recall, F1/Dice, IoU (Jaccard).

---

## 2) Part A‑1 — Baseline replication (Amazon 4‑band)

### 2.1 Data (Amazon 4‑band)
Follow the original repo’s instructions inside:
`baseline_replication/attention_unet/README.md`

After you download the Amazon dataset, put it under the expected folder layout (see that README), then preprocess:

```bash
cd baseline_replication/attention_unet
python preprocess-4band-amazon-data.py
# outputs: amazon-processed-large/{training,validation,test}/{images,masks}/*.npy
cd ../..
```

### 2.2 Run the replication notebook
Open and run:

- `baseline_replication/reproduce_baseline.ipynb`

This notebook trains both models and saves the best checkpoints under:

- `baseline_replication/checkpoints/unet-4d-amazon.h5`
- `baseline_replication/checkpoints/att-unet-4d-amazon.h5`

### 2.3 Reproduced baseline metrics (from executed notebook output)

Validation metrics printed by the notebook:

| Model (Amazon 4‑band) | Accuracy | Precision | Recall | F1/Dice | IoU |
|---|---:|---:|---:|---:|---:|
| U‑Net | **0.961452** | **0.975770** | **0.949722** | **0.962570** | **0.927840** |
| Attention U‑Net | **0.961147** | **0.980714** | **0.944119** | **0.962069** | **0.926910** |

The notebook also includes the paper-vs-replication table (Accuracy/Precision/Recall/F1) and a ±5% tolerance check.

---

## 3) Part A‑2 — New context: East Kalimantan (Indonesia)

### 3.1 SDG alignment
- **SDG 13 – Climate Action**
- **SDG 15 – Life on Land**

### 3.2 Why this is a meaningful transfer test
Compared with the Amazon, Kalimantan has different land‑use patterns (e.g., logging / plantations), spectral conditions, and potentially different error modes. This creates a realistic **domain shift** for transferability testing.

---

## 4) Part A‑3 — Contextual dataset (2023 Kalimantan)

All steps below are implemented in the local notebook:

- `local_context/contextual_adaptation_kalimantan_2023_en_updated_downloads_(1) (2).ipynb`

### 4.1 Task definition used in the notebook (IMPORTANT: aligns with outputs)
- **AOI (bbox):** `(116.0, -1.8, 117.2, -0.6)`
- **Year:** `2023` only (not multi-year)
- **Input:** Sentinel‑2 SR (harmonized) **2023 median composite**, 4 bands
- **Label:** Hansen GFC v1.12 **lossyear == 2023** (binary), resampled to the Sentinel‑2 grid using **nearest neighbor**

### 4.2 Expected input files
The notebook expects the following GeoTIFFs under:

`local_context/data_raw/`

- `S2_EKAL_2023_MEDIAN_4B_UTM.tif`  
  *(or multiple tiles named `S2_EKAL_2023_MEDIAN_4B_UTM-*.tif`, which will be mosaicked to `S2_EKAL_2023_MEDIAN_4B_UTM_FULL.tif`)*
- `HANSEN_LOSS_2023_UTM_30m.tif`

### 4.3 Getting the GeoTIFFs (GEE + fallback)
The notebook supports two ways:

1) **Google Earth Engine export (recommended for Sentinel‑2)**  
   Start exports, then download from Google Drive into `local_context/data_raw/`.

2) **No‑GEE fallback for Hansen**  
   Directly download the official 10×10° `lossyear` tile via HTTP, then clip/reproject to match Sentinel‑2.  
   (This path is included because some environments hit GEE permission/auth issues.)

> If your Earth Engine API fails (e.g., 403 / project disabled), you can still:  
> (a) export via Earth Engine Code Editor / Colab, then download from Drive;  
> (b) or use the notebook’s direct Hansen download for labels.

---

## 5) Part A‑4 — Adaptation choices (architecture + training pipeline)

### 5.1 Preprocessing + patching (exact values used)
- Raster alignment: resample Hansen 30m mask → Sentinel‑2 grid (nearest neighbor)
- **Patch size:** `PATCH = 512`
- **Stride (overlap):** `STRIDE = 256`
- Patch label stats:
  - `POS_THRESH = 0.005`  (patch considered “positive” if pos_rate ≥ 0.5%)
  - Keep all positive patches; keep negative patches with probability `KEEP_NEG_PROB = 0.5`
- **Leakage‑aware split:** `GroupShuffleSplit` using spatial blocks  
  `BLOCK = STRIDE * 4`, split by `block_id` (train/val/test).

### 5.2 Normalization (as implemented)
Mean/std are computed from **training patches only** via streaming (so we never load the whole dataset at once).  
From the executed notebook output in this run:

- `mean = [658.6302, 460.90695, 2755.205, 1641.4781]`
- `std  = [313.3643, 373.33643, 1398.8584, 949.323]`

*(If you re-export a different AOI or year, these numbers will change — but the procedure stays the same.)*

### 5.3 Model and losses (baseline vs adapted)
**Model backbone (both):** PyTorch U‑Net with `base=16`, `in_ch=4`, `out_ch=1`.

- **Baseline loss:** `BCEWithLogitsLoss()`
- **Adapted loss:** weighted BCE + soft Dice  
  - `pos_weight = clip((1-pos_rate)/pos_rate, 1.0, MAX_POS_W)` with `MAX_POS_W = 3.0`
  - `loss = ALPHA * BCE(pos_weight) + (1-ALPHA) * Dice`, with `ALPHA = 0.7`

### 5.4 Training hyperparameters (exact values used)
- `lr = 3e-4`
- `epochs = 8` (with early stopping)
- `patience = 5`
- `batch size = 2`
- Early stopping monitor: `IoU_mean_pos`

### 5.5 Threshold tuning (important)
The notebook **sweeps thresholds on the validation set** and picks the one maximizing `Dice_mean_pos`:

- Baseline best threshold: **0.15**
- Adapted best threshold: **0.40**

---

## 6) Part A‑5 — Evaluation + statistics + failure cases (Kalimantan 2023)

### 6.1 Test set metrics (from executed notebook output)
The test evaluation reports both:
- **Dice_mean_pos / IoU_mean_pos** (averaged over GT‑positive patches),
- and **global** Dice/IoU.

From the executed notebook output:

| Model | Threshold | Dice_mean_pos (CI95) | IoU_mean_pos | Dice_global | IoU_global | #GT‑positive patches |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.15 | **0.129820** (0.117158, 0.143309) | 0.073033 | 0.126100 | 0.067293 | 247 |
| Adapted | 0.40 | **0.176402** (0.157379, 0.195283) | 0.106079 | 0.162953 | 0.088704 | 247 |

The paired test set size used in Section 11 of the notebook:
- `N_all = 352` patches
- `N_pos = 247` GT‑positive patches

### 6.2 Statistical significance (paired per‑patch)
The notebook computes per‑patch metrics and runs paired tests on `(adapted − baseline)`:

**IoU per patch**
- ALL patches (N=352): mean diff **0.02319**, CI95 **(0.01662, 0.03096)**  
  t‑test p = **4.65e‑10**, Wilcoxon p = **8.06e‑28**
- GT‑positive patches (N=247): mean diff **0.03305**, CI95 **(0.02342, 0.04351)**  
  t‑test p = **2.92e‑10**, Wilcoxon p = **2.58e‑27**

**Dice per patch**
- ALL patches (N=352): mean diff **0.03269**, CI95 **(0.02442, 0.04144)**  
  t‑test p = **1.96e‑12**, Wilcoxon p = **5.24e‑28**
- GT‑positive patches (N=247): mean diff **0.04658**, CI95 **(0.03546, 0.05979)**  
  t‑test p = **8.93e‑13**, Wilcoxon p = **1.27e‑27**

### 6.3 Failure case analysis (where to find it)
The notebook includes qualitative plots comparing:
- Sentinel‑2 inputs
- GT loss mask
- baseline vs adapted predictions

Common failure modes discussed:
- confusion with bare soil / bright surfaces
- under‑segmentation of small fragmented loss
- boundary noise due to 30m→10m resampling

---

## 7) Environment setup

This repo mixes **TensorFlow/Keras** (baseline replication) and **PyTorch + raster tooling** (local context).
You can use one environment, but two environments are cleaner.

### Option A: single environment (quick)
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install -U pip
pip install -r requirements.txt

# local_context extras used by the notebook:
pip install torch scipy requests earthengine-api
```

### Option B: two environments (recommended)
- `baseline_replication`: TensorFlow/Keras
- `local_context`: PyTorch + raster + GEE

---

## 8) How to run (minimal checklist)

### Baseline replication
- [ ] Download Amazon dataset (see `baseline_replication/attention_unet/README.md`)
- [ ] Run preprocessing script to produce `amazon-processed-large/`
- [ ] Run `baseline_replication/reproduce_baseline.ipynb` end-to-end

### Kalimantan 2023
- [ ] Put the two GeoTIFFs into `local_context/data_raw/`
- [ ] Run `local_context/contextual_adaptation_kalimantan_2023_en_updated_downloads_(1) (2).ipynb` end-to-end

---

## 9) Notes on reproducibility

- Random seeds are set (`SEED=42`) in both notebooks.
- For Kalimantan, patch split is **spatially grouped** to reduce leakage.
- If you change AOI/year/export resolution, results will change; the code path is the same.

---

## 10) License
See `LICENSE`.
