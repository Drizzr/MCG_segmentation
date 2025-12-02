

# ECG & MCG Segmentation Framework

## Overview

This project provides a robust PyTorch-based framework for segmenting Electrocardiogram (ECG) and Magnetocardiogram (MCG) signals into their core cardiac cycle components: **No Wave, P-Wave, QRS Complex, and T-Wave**.

Designed to enable deep learning on **MCG data** (which lacks large open-source datasets), this framework leverages established ECG databases (QTDB, LUDB) for training. It addresses specific domain challenges:
- **Short-Window Analysis:** Effective on short segments (1-2s) vs standard 10s windows.
- **Signal Normalization:** Two-step normalization to bridge the amplitude gap between ECG and MCG.
- **Independent Channel Processing:** Treats channels independently to handle the directional components of MCG.

### Key Features
- **Data Sources:** Automates processing of **QT Database (QTDB)** and **Lobachevsky University Database (LUDB)** [2, 7, 8].
- **Architectures:** Implements **1D U-Nets with Multi-Head Self-Attention** [3], hybrid CNN-BiLSTM-Transformer models, and DENS-ECG [4].
- **Robust Augmentation:** Simulates physiological noise (baseline wander, EMG) and sensor noise.
- **Standardized Evaluation:** Adheres to **AAMI standards** (±150 ms tolerance) reporting Se, PPV, F1, and $m \pm \sigma$.

---

## 2. Project Structure

```
project_root/
├── Datasets/
│   ├── base/               # Raw scripts for QTDB/LUDB
│   ├── train/              # Generated Training CSVs
│   ├── val/                # Generated Validation CSVs
│   └── create_training_data.py
├── model/
│   ├── data_loader.py      # Dataset class & Augmentations
│   ├── model.py            # Model architectures
│   └── trainer.py          # Training loop, logging, focal loss
├── train.py                # Main training script
├── evaluate.py             # Evaluation script
└── trained_models/         # Checkpoints and logs
```

---

## 3. Installation and Data Setup

### 3.1. Dependencies
```bash
pip install -r requirements.txt
```

### 3.2. Data Preprocessing
The framework uses a pipeline to download, harmonize (to 250 Hz), and split data.

**Dataset Characteristics:**
| Data Source | # Recordings | Duration | Frequency | Leads | Annotations |
|-------------|--------------|--------------------|-----------|-------|--------------------------------------|
| **QTDB** [7] | 105 | ~30 seconds | 250 Hz | 2 | P, QRS, T offsets (T-onsets artificial) |
| **LUDB** [8] | 200 | 10 seconds | 500 Hz | 12 | P, QRS, T (onsets & offsets) |

**Execution Steps:**
1.  **QTDB:** `cd Datasets/base/qtdb && python create_dataset.py`
2.  **LUDB:** `cd Datasets/base/ludb && python create_dataset.py`
3.  **Split Data:** `cd ../.. && python create_training_data.py`

*Note: Data is split at the **patient level**. Test set contains 40% of LUDB patients (no QTDB). Train/Val mixes QTDB (95%/5%) and LUDB (55%/5%).*

---

## 4. Data Loading & Augmentation

Implemented in `model/data_loader.py`.

### 4.1. Augmentation Pipeline
Applied with a probability (default 80%) during training to ensure robustness:

1.  **Amplitude Scaling:** $x_{scaled}[t] = \alpha \cdot x[t]$ where $\alpha \sim U(0.9, 1.1)$.
2.  **Time Shifting:** Cyclic shifting $x_{shifted}[t] = x[(t - s) \pmod{T}]$.
3.  **Baseline Wander:** Adds low-frequency sinusoids to simulate drift.
    $x_{bw}[t] = x[t] + \sum A_k \sin(2\pi f_k t + \phi_k)$
4.  **Gaussian Noise:** $\epsilon_t \sim \mathcal{N}(0, \sigma^2)$.
5.  **High-Frequency Noise:** Simulates powerline/EMG artifacts (1-29 Hz mix).

### 4.2. Normalization
Critical for MCG generalization.
1.  **Zero-Mean:** $x_{zero}[t] = x_{aug}[t] - \mu$
2.  **Max-Absolute Scaling:** $x_{norm}[t] = \frac{x_{zero}[t]}{\max |x_{zero}| + \epsilon}$

<div align="center">
  <img src="https://github.com/user-attachments/assets/da91c08b-345b-404b-8ccc-1984da9f7444" alt="augmentation_result" width="100%">
  <br>
  <em>Augmentation pipeline: Raw signal vs. Augmented & Normalized signal.</em>
</div>

---

## 5. Model Architectures

Defined in `model/model.py`.

| Model Name | Params | Key Features |
|:---|:---|:---|
| **Unet-1D-15M** | ~15M | 4-level U-Net, 8-head Self-Attention, Dropout. |
| **Unet-1D-900k** | ~900k | **(Recommended)** 3-level U-Net, 4-head Self-Attention, efficient. |
| **MCG-Segmentator_s** | ~375k | Multi-scale Conv + BiLSTM + Transformer. |
| **MCG-Segmentator_xl** | ~1.3M | Larger capacity version of above. |
| **DENS-Model** [4] | ~1.4M | Conv1D + BiLSTM. |

**U-Net Architecture:**
<div align="center">
  <img src="https://github.com/user-attachments/assets/6efcd9ba-54be-438d-9627-da8c27e2bcbe" alt="UNet Architecture" width="90%">
</div>

**MCG-Segmentator Architecture:**
<div align="center">
  <img src="https://github.com/user-attachments/assets/a6500515-d7b1-465f-85a0-fced5c35e874" alt="MCG Segmentator Architecture" width="90%">
</div>

---

## 6. Training

Managed by `trainer.py` using **Focal Loss** [6] (alpha=0.25, gamma=2.0) and **AdamW** [5].

### Usage
```bash
python train.py \
    --model_name Unet-1D-900k \
    --data_dir_train Datasets/train \
    --save_dir trained_models/Unet-1D-900k/checkpoints \
    --num_epochs 100 --batch_size 64 --sequence_length 500 --overlap 400
```

<div align="center">
  <table border="0" cellspacing="0" cellpadding="0">
    <tr>
      <td align="center"><img src="https://github.com/user-attachments/assets/943e5f4a-0d0a-41c2-98b1-42d1544a8daf" alt="accuracy" width="95%"></td>
      <td align="center"><img src="https://github.com/user-attachments/assets/dce211c5-c293-4279-ace8-9f79d56025d6" alt="lr_scheduler" width="95%"></td>
    </tr>
  </table>
  <br>
  <em>(a) Training/Validation accuracy. (b) Cosine Annealing Learning Rate schedule.</em>
</div>

---

## 7. Evaluation & Results

Evaluation (`evaluate.py`) follows **AAMI standards**:
- **Tolerance:** ±150 ms.
- **Post-processing:** Segments <40ms are removed.

### Usage
```bash
python evaluate.py --load_dir trained_models/Unet-1D-900k --data_dir_eval Datasets/test
```

### 7.1. Performance Summary
Delineation performance on the **LUDB test set**.

| Model | Metric | P onset | P offset | QRS onset | QRS offset | T onset | T offset |
|:--- |:--- |:---:|:---:|:---:|:---:|:---:|:---:|
| **Kalyakulina [8]** | F1 (%) | 97.42 | 97.42 | 99.74 | 99.74 | – | 98.43 |
| | m±σ (ms) | -2.7±10 | 0.4±11 | -8.1±7 | 3.8±8 | – | 5.7±15 |
| **Sereda [10]** | F1 (%) | 88.49 | 88.53 | 98.84 | 98.72 | 96.35 | 96.24 |
| **Moskalenko [11]**| F1 (%) | 96.47 | 96.43 | 99.90 | 99.90 | 99.15 | 99.21 |
| **Joung [12]** | F1 (%) | 97.24 | 97.27 | 99.48 | 99.78 | 99.74 | 99.52 |
| **DENS-Model [4]** | F1 (%) | 90.69 | 90.86 | 97.47 | 97.86 | 97.62 | 97.58 |
| | m±σ (ms) | -14±28 | 0.7±23 | 1.2±10 | -2.9±14 | -25±38 | 10±33 |
| **U-Net-1D-900k** | **F1 (%)** | **95.23** | **95.25** | **99.53** | **99.67** | **98.08** | **97.91** |
| (This work) | m±σ (ms) | 2.5±17 | 1.3±16 | 0.2±7 | -1.2±10 | -0.3±26 | -0.8±24 |
| **U-Net-1D-15M** | **F1 (%)** | **96.11** | **96.11** | **99.42** | **99.56** | **98.95** | **98.75** |
| (This work) | m±σ (ms) | -0.2±17 | 2.5±16 | 1.0±7 | -1.5±10 | -0.5±26 | 3.2±23 |
| **MCG Seg_s** | **F1 (%)** | 92.69 | 92.74 | 98.15 | 98.52 | 97.60 | 97.36 |
| (This work) | m±σ (ms) | -8.2±25 | 1.5±21 | 1.4±11 | -3.6±14 | -19±36 | 11±30 |
| **MCG Seg_xl** | **F1 (%)** | 92.77 | 92.80 | 98.13 | 98.54 | 97.50 | 97.28 |

### 7.2. Discussion
- **Efficiency:** The 900k parameter model performs comparably to the 15M model and state-of-the-art, validating the efficiency of the attention bottleneck.
- **P-Wave Bias:** Performance on P-waves is slightly lower than pure ECG models. This is a known trade-off from training on QTDB (where P-waves are ubiquitous) to ensure robustness for MCG, leading to some over-prediction on arrhythmic LUDB samples.

### 7.3. Visualization

**General Performance (Test Set Sample):**
<div align="center">
  <img  width="85%" src="https://github.com/user-attachments/assets/f071c2bf-fa0d-498e-bcbe-e0b318f436e5">
  <br>
  <em>Accurate detection of P, QRS, and inverted T-wave.</em>
</div>

**Handling High-Frequency Artifacts:**
<div align="center">
  <img src="https://github.com/user-attachments/assets/e28cfee9-f8db-44f3-9681-8abfa5e51d85" alt="Artifact Performance" width="85%">
  <br>
  <em>The model remains robust against sharp artifacts typical in LUDB/MCG.</em>
</div>

**Application to Noisy MCG (Key Result):**
<div align="center">
  <img src="https://github.com/user-attachments/assets/8e15773b-631a-43b0-8128-89116f7e0a55" alt="MCG Performance" width="85%">
  <br>
  <em>Successful segmentation of a <b>noisy single-channel MCG signal</b> despite high noise levels.</em>
</div>

---

## 9. Applications to Magnetocardiography

The model enables downstream tasks critical for MCG analysis:

1.  **Robust Peak Detection:**
    By restricting R-peak search to the predicted QRS interval, false positives from noise spikes are eliminated.
    <div align="center">
      <img width="90%" alt="qrs_detection" src="https://github.com/user-attachments/assets/19ac67c6-d7c7-4f57-b141-4a4e920db735" />
    </div>

2.  **Heartbeat Score:**
    A metric to automate channel selection in multi-lead MCG by combining:
    *   **Prediction Confidence:** Average softmax probability.
    *   **Physiological Plausibility:** Verifying segment ratios (e.g., P-wave duration vs QRS).

3.  **ICA Filtering:**
    Using the Heartbeat Score to identify and remove noise components in Independent Component Analysis (ICA). (See: [Bachelorarbeit](https://github.com/Drizzr/Bachelorarbeit)).

---

## 10. References

[1] Sattar, Y., & Chhabra, L. (2023). Electrocardiogram. *StatPearls*.
[2] Goldberger, A. L., et al. (2000). PhysioBank... *Circulation*.
[3] Vaswani, A., et al. (2017). Attention Is All You Need.
[4] Peimankar, A., & Puthusserypady, S. (2021). DENS-ECG. *Expert Systems with Applications*.
[5] Kingma, D. P., & Ba, J. (2017). Adam: A Method for Stochastic Optimization.
[6] Lin, T.-Y., et al. (2017). Focal Loss for Dense Object Detection.
[7] Laguna, P., et al. (1997). QT Database... *Computers in Cardiology*.
[8] Kalyakulina, A. I., et al. (2020). LUDB... *IEEE Access*.
[9] Kalyakulina, A., et al. (2021). LUDB (version 1.0.1). *PhysioNet*.
[10] Sereda, I., et al. (2018). ECG Segmentation by Neural Networks.
[11] Moskalenko, V., et al. (2020). Deep Learning for ECG Segmentation.
[12] Joung, C., et al. (2024). Deep learning based ECG segmentation.
[13] Kanhe, R. K., & Hamde, S. T. (2014). Wavelet-based compression of ECG signals.