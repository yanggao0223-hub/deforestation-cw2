# Deforestation Detection (CW2)

This repository is my **Coursework 2 (50%)** submission. It has two parts:

1) **Baseline replication (Part A‑1)** — reproduce the published deforestation segmentation baseline on the **Amazon 4‑band** dataset using **U‑Net** and **Attention U‑Net**.

2) **Contextual adaptation (Part A‑2 → A‑5)** — adapt the same segmentation approach to a **new geographic context**:
**East Kalimantan (Borneo), Indonesia**, using **2023** Sentinel‑2 imagery and **Hansen GFC `lossyear==2023`** supervision.

> Large datasets are **not** committed to this repo. This README documents how to place/download/export them so the notebooks run end‑to‑end.

---

## Results (from the executed notebooks in this repo)

### A‑1) Baseline replication — Amazon (4‑band), validation set
The table below is **exactly aligned** with the printed metrics in
`baseline_replication/reproduce_baseline.ipynb` (Section “Quantitative Evaluation on Validation Set”).

| Model (4‑band Amazon) | Source | Accuracy | Precision | Recall | F1 / Dice | IoU |
|---|---|---:|---:|---:|---:|---:|
| U‑Net | Reported (paper) | 0.9395 | 0.9766 | 0.9395 | 0.9441 | — |
| U‑Net | **Our run** | **0.9615** | **0.9758** | **0.9497** | **0.9626** | **0.9278** |
| Attention U‑Net | Reported (paper) | 0.9448 | 0.9790 | 0.9448 | 0.9459 | — |
| Attention U‑Net | **Our run** | **0.9611** | **0.9807** | **0.9441** | **0.9621** | **0.9269** |

✅ All reproduced metrics are within the typical **±5% tolerance** check implemented in the notebook.

---

### A‑2 → A‑5) Local context — East Kalimantan (2023), test set
Local notebook: `local_context/contextual_adaptation_kalimantan_2023_en_updated_downloads_(1) (2).ipynb`

We compare **two U‑Net training regimes** on the *same* 2023 setup:

- **U‑Net baseline**: direct transfer of the standard U‑Net training recipe to Kalimantan.
- **U‑Net adapted**: context‑motivated changes for Kalimantan (year alignment, imbalance handling, threshold tuning, etc.).

Thresholds are selected on the **validation set** by maximizing **positive‑class Dice**, then evaluated on the **test set**.

| Model (Kalimantan 2023) | Threshold (picked on val) | Dice (GT‑positive patches, test) | IoU (GT‑positive patches, test) | Dice_global (test) | IoU_global (test) | #GT‑positive test patches |
|---|---:|---:|---:|---:|---:|---:|
| U‑Net **baseline** | 0.15 | 0.1298 (95% CI: 0.1172–0.1433) | 0.0730 | 0.1261 | 0.0673 | 247 |
| U‑Net **adapted** | 0.40 | 0.1764 (95% CI: 0.1574–0.1953) | 0.1061 | 0.1630 | 0.0887 | 247 |

**Paired per‑patch significance tests (adapted − baseline, GT‑positive patches only):**
- mean ΔIoU = 0.0330, t‑test p = 2.92e‑10, Wilcoxon p = 2.58e‑27  
- mean ΔDice = 0.0466, t‑test p = 8.93e‑13, Wilcoxon p = 1.27e‑27

> Checkpoints included for reproducibility:  
> `local_context/unet_baseline_best.pt`, `local_context/unet_adapted_best.pt`

---

## Repository structure

```
.
├── baseline_replication/
│   ├── reproduce_baseline.ipynb
│   └── attention_unet/                 # reference implementation + preprocess scripts
├── local_context/
│   ├── contextual_adaptation_kalimantan_2023_en_updated_downloads_(1) (2).ipynb
│   ├── unet_baseline_best.pt
│   └── unet_adapted_best.pt
├── requirements.txt
└── LICENSE
```

---

## Setup

### Recommended: two environments (TF vs Torch)
The baseline notebook uses **TensorFlow/Keras**, while the local context notebook uses **PyTorch + raster tooling**.

**Baseline env (TensorFlow)**
```bash
python -m venv .venv_tf
source .venv_tf/bin/activate  # Windows: .venv_tf\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

**Local env (PyTorch)**
```bash
python -m venv .venv_torch
source .venv_torch/bin/activate  # Windows: .venv_torch\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install torch scipy earthengine-api
```

> If `rasterio` / `rioxarray` fails to install locally (GDAL issues), run the local notebook on **Google Colab**.

---

## Part A‑1 — Baseline replication (Amazon 4‑band)

### A‑1.1 Download the dataset (Zenodo)
Follow the dataset instructions inside:
`baseline_replication/attention_unet/README.md`

Place the Amazon dataset under:
`baseline_replication/attention_unet/AMAZON/`

### A‑1.2 Preprocess into `.npy` patches
```bash
cd baseline_replication/attention_unet
python preprocess-4band-amazon-data.py
# outputs to: amazon-processed-large/...
```

### A‑1.3 Run the replication notebook
Open and run:
- `baseline_replication/reproduce_baseline.ipynb`

This notebook:
- prints environment info + GPU availability
- sets a fixed seed (`SEED = 42`)
- trains U‑Net and Attention U‑Net
- evaluates on the validation set (Accuracy/Precision/Recall/F1/IoU)
- compares to the paper’s reported baselines (±5% check)

---

## Part A‑2 → A‑5 — Local context (East Kalimantan 2023)

### A‑2.1 Task definition
**Binary segmentation** of 2023 forest loss:

- **Input**: Sentinel‑2 SR (harmonized) **2023 median composite**, 4 bands (G/R/NIR/SWIR)
- **Label**: Hansen GFC `lossyear == 2023` (optionally masked by `treecover2000`)
- **Splits**: patch‑based train/val/test, with seed control (`SEED = 42`)

### A‑2.2 Data export (GeoTIFFs)
The notebook supports Earth Engine export logic, but if Earth Engine auth is blocked in your environment:

1) Export in **Earth Engine Code Editor** or **Colab**
2) Download to your machine / Drive
3) Put files here:
`local_context/data_raw/`

Expected filenames (or update the paths in the notebook):
- `S2_EKAL_2023_MEDIAN_4B_UTM.tif`
- `HANSEN_LOSS_2023_UTM_30m.tif`

### A‑3/A‑4 Contextual adaptation choices (implemented in the notebook)
Main motivations for Kalimantan:
- severe **class imbalance**
- label/input **resolution mismatch** (Hansen 30m vs S2 10m)
- **domain shift** vs Amazon (spectral + land cover differences)

Adaptation implements:
- strict **2023↔2023** alignment (input year matches label year)
- projection/resampling controls for aligned tiles
- imbalance‑aware loss / sampling
- **threshold tuning on validation** (do not hard‑code 0.5)
- evaluation on per‑patch distributions + bootstrap CIs + paired significance tests
- qualitative **failure case** visualizations

---

## Part B — Poster & 2‑minute presentation
Poster should cover:
1) the original baseline paper + method  
2) the new context + dataset + why it matters  
3) results + limitations + ethical considerations  
4) link to this GitHub repo

---

## References
- John, D., & Zhang, C. (2022). *An attention-based U-Net for detecting deforestation within satellite sensor imagery.*
- ESA Sentinel‑2 Level‑2A Surface Reflectance (harmonized)
- Hansen Global Forest Change (GFC)
- Zenodo Amazon/Atlantic Forest datasets (see `baseline_replication/attention_unet/README.md`)
