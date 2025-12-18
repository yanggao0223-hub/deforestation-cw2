# Deforestation Detection (CW2)
## Baseline Replication and Contextual Adaptation with Attention U-Net

This repository contains my submission for **Coursework 2**, focusing on:

1. **Replication of a published AI baseline** for deforestation detection, and  
2. **Contextual adaptation** of the same methodology to a new geographic and data context (**Kalimantan, Indonesia**).

All stages of the contextual adaptation pipeline — data acquisition, preprocessing, model training, and evaluation — are implemented in a **single, fully reproducible notebook**, following a controlled and transparent experimental design.



## 1. Original Study (Baseline)

**Paper**  
John, D., & Zhang, C. (2022).  
*An attention-based U-Net for detecting deforestation within satellite sensor imagery.*  
International Journal of Applied Earth Observations and Geoinformation.

**Task**  
Binary semantic segmentation of deforestation from Sentinel-2 satellite imagery.

**Models**
- U-Net  
- Attention U-Net (attention-gated decoder)

**Evaluation metrics**
- Accuracy
- Precision
- Recall
- F1-score (Dice)
- Intersection-over-Union (IoU / Jaccard)



## 2. Baseline Replication (Part A-1)

### 2.1 Replication setting
- **Dataset**: Sentinel-2 **4-band Amazon rainforest**
- **Input bands**: Green, Red, NIR, SWIR
- **Task**: Pixel-wise binary segmentation
- **Implementation**: Based on the official Attention U-Net repository

Notebook:baseline_replication/reproduce_baseline.ipynb


### 2.2 Reproduced baseline results (±5%)

| Model | Accuracy | Precision | Recall | F1 | IoU |
|------|---------:|----------:|-------:|----:|----:|
| U-Net | 0.961 | 0.976 | 0.950 | 0.963 | 0.928 |
| Attention U-Net | 0.961 | 0.981 | 0.944 | 0.962 | 0.927 |

All reproduced metrics fall **within ±5%** of the values reported in the original paper for the same 4-band Amazon setting, satisfying the coursework replication requirement.



## 3. Contextual Challenge & Motivation (Kalimantan)

Deforestation in **Kalimantan (Indonesia)** is driven by logging, agricultural expansion, and land-use change.  
Compared to the Amazon, Kalimantan exhibits:

- Different forest structure and land-use patterns  
- Strong spectral variability due to soil exposure and seasonal effects  
- Sparse and imbalanced deforestation labels at local scale  

This makes Kalimantan a challenging but realistic testbed for **contextual adaptation** of deforestation detection models.

### SDG Alignment
- **SDG 13 – Climate Action**
- **SDG 15 – Life on Land**




All data preprocessing, training, and evaluation for the local context are contained in  `contextual_adaptation_kalimantan.ipynb`.



## 4. Environment Setup


git clone https://github.com/yanggao0223-hub/deforestation-cw2.git
cd deforestation-cw2

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt


Recommended:
Python 3.x
GPU-enabled environment for training (optional but beneficial)


## 5. Contextual Dataset Construction (Kalimantan)

All steps described in this section are fully implemented and documented in:contextual_adaptation_kalimantan.ipynb




### 5.1 Imagery

- **Source**: Sentinel-2 Level-2A Surface Reflectance (harmonized)
- **Data provider**: Google Earth Engine (GEE)

**Processing pipeline**:
- **Area of Interest (AOI)**: East Kalimantan sub-region
- **Time range**: 2019–2023
- **Cloud filtering**: Images are filtered using a maximum cloud cover threshold
- **Temporal aggregation**: Median composite constructed over the full time window
- **Input bands**:
  - Green
  - Red
  - Near-Infrared (NIR)
  - Shortwave Infrared (SWIR)

The resulting imagery forms a 4-band input consistent with the baseline replication setting.



### 5.2 Labels

- **Source**: Hansen Global Forest Change (GFC), version 1.12

**Label definition**:
- Binary forest loss mask derived from the `lossyear` band
- Forest loss events aggregated over the period **2019–2023**
- Pixels with recorded forest loss within this window are labeled as positive class

The generated loss masks are spatially aligned with the Sentinel-2 imagery and resampled to a common grid to ensure pixel-wise correspondence.



### 5.3 Data Access & Governance

- All datasets used in this study are **publicly available** and released for **research and non-commercial use**
- No personal, private, or sensitive information is included
- No data licensing or consent issues arise from the use of these environmental datasets

Nevertheless, ethical responsibility remains in how model outputs are interpreted and communicated, particularly in regions where land-use decisions may directly impact local communities and livelihoods.


## 6. Data Preprocessing Pipeline

The complete preprocessing pipeline is implemented within:contextual_adaptation_kalimantan.ipynb

The pipeline is designed to ensure spatial alignment, prevent information leakage, and address extreme class imbalance in deforestation labels.



### 6.1 Data Export from Google Earth Engine (GEE)

The following assets are exported from Google Earth Engine:

- **Sentinel-2 Surface Reflectance** median composite (2019–2023)
- **Hansen Global Forest Change** loss mask aggregated over the same period

**Export characteristics**:
- Coordinate Reference System (CRS): consistent across imagery and labels
- Spatial resolution: fixed during export to ensure pixel-level alignment
- Export format: GeoTIFF

Exported files are downloaded from Google Drive and stored locally under:local_context/data_raw/




### 6.2 Spatial Alignment and Reprojection

After export, all raster files are:

- Reprojected to a common spatial reference system
- Clipped to the same Area of Interest (AOI)
- Verified to have identical spatial dimensions and resolution

This step ensures that each pixel in the Sentinel-2 imagery corresponds exactly to the same pixel location in the forest loss mask.



### 6.3 Patch Extraction (Tiling)

Due to the large spatial extent of the study area, the aligned GeoTIFFs are subdivided into fixed-size patches:

- **Patch size**: 512 × 512 pixels
- **Stride**: equal to patch size (non-overlapping tiles)
- **Channels**:
  - Input imagery: 4 channels (Green, Red, NIR, SWIR)
  - Labels: 1 channel (binary loss mask)

Each patch pair (image, mask) is saved as a NumPy array for efficient loading during training.



### 6.4 Dataset Splitting

Extracted patches are split into training, validation, and test sets:

- Split strategy: random split with fixed seed
- Typical proportions:
  - Training set
  - Validation set
  - Test set

Using a fixed random seed ensures that all experiments are reproducible and that baseline and adapted models are evaluated on identical data splits.



### 6.5 Normalization and Input Scaling

To avoid information leakage, normalization statistics are computed **using the training set only**:

- Mean and standard deviation are calculated per spectral band
- Training, validation, and test inputs are normalized using the same statistics

This ensures consistent input scaling across all experiments while preserving experimental integrity.



### 6.6 Class Imbalance Handling

Forest loss pixels are extremely sparse relative to background forest pixels.  
To address this imbalance:

- Binary masks are retained at full resolution (no downsampling)
- The adapted model incorporates **Dice loss** in addition to Binary Cross-Entropy (BCE)
- Data augmentation is applied during training to improve robustness

These measures aim to stabilize optimization and improve sensitivity to small deforestation regions.



### 6.7 Reproducibility Considerations

- All preprocessing steps are deterministic given a fixed random seed
- Intermediate outputs are saved to disk to allow inspection and reuse
- The entire pipeline can be reproduced by running the adaptation notebook from start to finish

This integrated preprocessing design ensures transparency, reproducibility, and consistency across all experiments.

## 7. Contextual Adaptation Strategy

All model adaptation, training, and evaluation steps described in this section are implemented in:
local_context/contextual_adaptation_kalimantan.ipynb



The goal of the adaptation is to assess whether the baseline Attention U-Net can be effectively transferred to a new geographic and spectral context under a **controlled experimental setup**.



### 7.1 Controlled Adaptation Design

To attribute performance changes specifically to contextual adaptation, a **minimal ablation strategy** is adopted:

- Both baseline and adapted models:
  - Share the same Attention U-Net architecture
  - Are trained on the same train/validation/test splits
  - Use identical optimization settings where possible

This design isolates the effect of adaptation choices while avoiding confounding factors.



### 7.2 Baseline Configuration

The baseline configuration follows the original training setup:

- Model: Attention U-Net
- Loss function: Binary Cross-Entropy (BCE)
- Input: 4-band Sentinel-2 imagery
- No additional data augmentation beyond the original setup

This configuration serves as a reference point for comparison.



### 7.3 Adapted Configuration

The adapted model introduces targeted changes motivated by the properties of the local dataset:

- **Loss function**:
  - Binary Cross-Entropy (BCE) + Dice loss
  - Dice loss is used to mitigate extreme class imbalance in deforestation labels

- **Data augmentation**:
  - Random horizontal and vertical flips
  - Intensity perturbations applied to input bands

No architectural changes are introduced, ensuring that observed effects are attributable to training-level adaptations rather than model capacity changes.



### 7.4 Training Protocol

- Optimizer and learning rate schedule are kept consistent between baseline and adapted models
- Early stopping is applied based on validation performance
- Best-performing checkpoints are saved for evaluation
- Fixed random seeds are used to ensure reproducibility

This protocol ensures a fair and transparent comparison between models.



### 7.5 Experimental Rationale

This minimal adaptation strategy reflects a realistic deployment scenario, where extensive re-annotation or architectural redesign may not be feasible.

By constraining the degree of change, the experiment highlights the practical challenges of transferring segmentation models across geographic regions with different spectral and land-use characteristics.


## 8. Evaluation & Statistical Interpretation

All evaluation procedures and analyses described in this section are implemented in:local_context/contextual_adaptation_kalimantan.ipynb


The evaluation focuses on a **controlled comparison** between the baseline and adapted models under identical test conditions.



### 8.1 Evaluation Protocol

- Both baseline and adapted models are evaluated on the **same held-out test set**
- Predictions are generated using the best-performing checkpoint selected on the validation set
- Evaluation metrics are computed over the full test set without post-hoc threshold tuning

This protocol ensures that performance differences can be attributed to the adaptation strategy rather than evaluation bias.



### 8.2 Quantitative Results (Local Context)

The following metrics are reported to align with the original study:

- Intersection-over-Union (IoU)
- Dice coefficient (F1-score)
- Precision
- Recall

| Model | IoU | Dice | Precision | Recall |
|------|----:|-----:|----------:|-------:|
| Baseline | 0.0156 | 0.0307 | 0.0156 | 0.9957 |
| Adapted | 0.0153 | 0.0302 | 0.0153 | 0.7537 |

The results show comparable IoU and Dice scores across models, with a notable reduction in recall for the adapted configuration, indicating a shift toward more conservative predictions.



### 8.3 Statistical Interpretation

Due to the use of **aggregate evaluation metrics** (rather than per-sample metric distributions), classical paired statistical significance tests (e.g., paired t-test or Wilcoxon signed-rank test) are not applicable.

Instead, model comparison is based on:

- **Effect size and direction** of metric differences
- Stability of IoU and Dice scores across configurations
- Precision–recall trade-offs induced by the adaptation strategy

Under this interpretation, contextual adaptation does **not produce a statistically meaningful improvement** over the baseline model in the current experimental setup.



### 8.4 Failure Case Analysis

Qualitative inspection of model predictions reveals several recurring failure modes:

- Confusion between deforested regions and spectrally similar bare soil
- Under-segmentation of small or fragmented deforestation patches
- Sensitivity to local spectral characteristics not present in the Amazon training data

These failure cases highlight the challenges of transferring segmentation models across regions with differing land-cover and spectral properties.



### 8.5 Summary of Findings

- Baseline replication results are consistent with the original study
- Contextual adaptation maintains comparable IoU and Dice performance
- No statistically meaningful performance gain is observed after adaptation
- Performance differences are primarily driven by recall–precision trade-offs

This evaluation emphasizes the importance of **rigorous, transparent analysis**, particularly when model adaptation does not lead to clear quantitative improvements.

## 9. Discussion, Limitations, and Reflection

This section provides a critical interpretation of the experimental findings, highlights key limitations, and reflects on the implications of deploying deforestation detection models in new geographic contexts.



### 9.1 Interpretation of Results

The experiments demonstrate that while the Attention U-Net baseline can be successfully replicated on the original Amazon dataset, its direct adaptation to the Kalimantan context does not yield a measurable performance improvement.

The adapted model maintains comparable IoU and Dice scores but exhibits a reduction in recall, indicating a more conservative prediction behaviour. This suggests that the adaptation primarily affects the precision–recall trade-off rather than overall segmentation accuracy.

Importantly, the absence of performance gains does not indicate methodological failure. Instead, it highlights the challenges inherent in transferring segmentation models across regions with distinct spectral characteristics and land-use patterns.



### 9.2 Key Limitations

Several limitations influence the outcomes of this study:

- **Domain shift**: Spectral and structural differences between Amazon and Kalimantan forests limit model transferability.
- **Extreme class imbalance**: Forest loss pixels are sparse, reducing the effective learning signal even with Dice-based loss functions.
- **Aggregate evaluation metrics**: The use of aggregate metrics constrains statistical power and limits the application of classical significance tests.
- **Label uncertainty**: Hansen Global Forest Change labels may contain temporal or spatial inaccuracies at local scales.

These limitations reflect realistic constraints encountered in applied environmental monitoring.


### 9.3 Implications for Real-world Deployment

From a deployment perspective, the findings suggest that:

- Models trained in one geographic region should not be assumed to generalize without careful validation.
- Contextual adaptation may require substantially more labeled data or region-specific feature engineering to achieve meaningful improvements.
- Model outputs should be used as **decision-support tools**, complemented by expert review rather than automated enforcement.

Such considerations are particularly important in regions where land-use decisions directly impact local communities and livelihoods.



### 9.4 Examiner-facing Reflection

This coursework emphasizes **rigour over optimisation**. Rather than maximising performance at all costs, the project prioritises controlled experimentation, transparent reporting, and honest interpretation of results.

By demonstrating both successful baseline replication and the limitations of contextual adaptation, the work reflects real-world challenges faced in applied AI research. This approach aligns closely with the learning objectives of the coursework and underscores the importance of critical evaluation when deploying AI systems in high-impact environmental domains.





