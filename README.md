# MedCLIP-TB

A contrastive vision-language model for zero-shot tuberculosis detection in chest radiographs, built on the CLIP framework using PyTorch.

---

## Overview

MedCLIP-TB trains a joint image-text embedding space by aligning frontal chest X-ray images with paired radiology report captions. Without any explicit classification supervision, the model learns to distinguish normal from tuberculosis cases purely through contrastive pretraining — a zero-shot capability that emerges from the alignment of visual and linguistic representations.

The model is trained on 800 frontal chest radiographs from the Montgomery County and Shenzhen Hospital tuberculosis screening datasets, achieving 94% zero-shot classification accuracy on the binary normal vs. TB detection task.

---

## Results

### Zero-Shot Classification

| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Normal       | 0.93      | 0.95   | 0.94     | 406     |
| Tuberculosis | 0.94      | 0.93   | 0.94     | 394     |
| **Overall**  |           |        | **0.94** | **800** |

### Cross-Modal Retrieval

| Metric       | Score  |
|--------------|--------|
| Recall@1     | 0.119  |
| Recall@5     | 0.393  |
| Recall@10    | 0.585  |
| MRR          | 0.272  |
| Median Rank  | 8      |

---

## Project Structure

```
MEDICAL_VLM/
├── src/
│   ├── model.py          # MedicalCLIPModel architecture
│   ├── dataset.py        # MedicalVLMDataset loader
│   ├── train.py          # Training loop
│   └── evaluate.py       # Retrieval and classification evaluation
├── scripts/
│   └── build_metadata.py # Builds metadata.csv from raw datasets
├── data/
│   ├── raw/
│   │   ├── MontgomerySet/
│   │   └── ChinaSet/
│   └── processed/
│       └── metadata.csv
└── outputs/
    └── checkpoints/
        └── medical_clip_final.pt
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/MEDICAL_VLM.git
cd MEDICAL_VLM
```

### 2. Create and activate environment
```bash
conda create -n medicalvlm python=3.11
conda activate medicalvlm
```

### 3. Install dependencies
```bash
pip install torch torchvision transformers pandas pillow scikit-learn seaborn matplotlib
```

---

## Datasets

Download both datasets and place them in `data/raw/`:

- **Montgomery County CXR Dataset** — 138 frontal chest X-rays screened for TB, with clinical reading text files
  - Download: https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip
  - Place at: `data/raw/MontgomerySet/`

- **Shenzhen Hospital CXR Dataset** — 662 frontal chest X-rays with TB labels and metadata
  - Download: https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-shenzhen
  - Place at: `data/raw/ChinaSet/`

---

## How to Run

### Step 1 — Build metadata CSV
Parses both datasets and generates descriptive image-text pairs:
```bash
python -m scripts.build_metadata
```

### Step 2 — Train the model
```bash
python -m src.train
```
Training runs for 15 epochs with AdamW optimizer and OneCycleLR scheduler. Checkpoint is saved to `outputs/checkpoints/`.

### Step 3 — Evaluate
```bash
python -m src.evaluate
```
Outputs retrieval metrics, classification report, and saves confusion matrix to `outputs/confusion_matrix.png`.

---

## Model Architecture

| Component        | Details                                      |
|------------------|----------------------------------------------|
| Image Encoder    | ResNet-18 pretrained on ImageNet (512-dim)   |
| Text Encoder     | DistilBERT-base-uncased (768-dim)            |
| Embedding Space  | Shared 256-dimensional projection            |
| Loss Function    | Symmetric cross-entropy contrastive loss     |
| Optimizer        | AdamW (lr=5e-5, weight decay=0.01)           |
| Scheduler        | OneCycleLR with linear warmup                |

---

## Training on Google Colab

For GPU-accelerated training, use Google Colab with a T4 GPU. Mount your Google Drive, copy data to local disk, then train:

```python
%cd /content/MEDICAL_VLM
!python -m src.train
```

Training time: ~2-3 minutes on T4 GPU vs ~30 minutes on CPU.

---

## Requirements

- Python 3.11
- PyTorch >= 2.0
- torchvision
- transformers
- pandas
- Pillow
- scikit-learn
- seaborn
- matplotlib

---

## Acknowledgements

- Montgomery County CXR dataset: Jaeger et al., 2014
- Shenzhen Hospital CXR dataset: Jaeger et al., 2014
- CLIP framework: Radford et al., 2021 — Learning Transferable Visual Models From Natural Language Supervision
- DistilBERT: Sanh et al., 2019

---

## License

This project is intended for research and educational purposes only. The datasets are subject to their respective licensing terms from the NIH and Kaggle sources listed above.
