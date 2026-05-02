# Continual Learning on Split-CIFAR-100
**CS514 Project — University of Illinois Chicago**

Exploring catastrophic forgetting in class-incremental learning without any replay buffer. We implement and compare eight methods, from naive fine-tuning to a frozen ViT-B/16 backbone with per-task adapters and NCM classification.

---

## Results Summary

| Method | Backbone | Classifier | AA (%) | BWT (%) |
|---|---|---|---|---|
| Naive Fine-Tuning | ResNet-18 | Linear Head | 12.86 | -60.95 |
| LwF | ResNet-18 | Linear Head | 16.58 | -56.09 |
| Hybrid (EWC + LwF) | ResNet-18 | Linear Head | 17.56 | -43.24 |
| EWC | ResNet-18 | Linear Head | 21.73 | -40.66 |
| ViT Naive | ViT-B/16 | Linear Head | 36.32 | -69.58 |
| ViT LwF | ViT-B/16 | Linear Head | 36.15 | -69.62 |
| ViT EWC | ViT-B/16 | Linear Head | 41.14 | -62.88 |
| ADAM (ResNet-18) | ResNet-18 | NCM | 47.63 | -15.54 |
| PASS (ResNet-18) | ResNet-18 | NCM | 57.20 | -10.79 |
| **ADAM (ViT-B/16)** | **ViT-B/16** | **NCM** | **74.81** | **-9.64** |

---

## Branches

- **`main`** — ViT-based methods, PASS, ADAM, CORe50 evaluation, report
- **`yuvan-code`** — First four ResNet-18 methods: Naive, LwF, EWC, Hybrid (EWC + LwF)

---

## Project Structure

```
├── methods/
│   ├── pass_resnet.py       # PASS: prototype augmentation + NCM
│   └── adam_resnet.py       # ADAM on ResNet-18 backbone
├── core50_eval/
│   └── run_core50.py        # Evaluates Naive + ADAM on CORe50
├── results/                 # JSON result files (one per method)
├── report.tex               # Final project report (ICLR format)
├── plot_report.py           # Generates all report figures
├── plot_adam_flowchart.py   # ADAM architecture diagram
├── plot_vit_adapter.py      # ViT methods comparison bar chart
├── save_notebook_results.py # Seeds results/ from notebook runs
└── requirements.txt
```

---

## Setup

```bash
conda create -n cs514 python=3.10
conda activate cs514
pip install -r requirements.txt
```

---

## Running Methods

**PASS (ResNet-18):**
```bash
python methods/pass_resnet.py
```

**ADAM (ResNet-18):**
```bash
python methods/adam_resnet.py
```

**ADAM (ViT-B/16) — best method:**  
Run from the ViT adapter notebook or script in the root directory.

**CORe50 evaluation:**
```bash
# First run (downloads ~3.8GB dataset):
python core50_eval/run_core50.py --download

# Subsequent runs:
python core50_eval/run_core50.py
```

**Generate report figures:**
```bash
python plot_report.py
```

---

## Datasets

- **Split-CIFAR-100:** Downloaded automatically via `torchvision` to `data/`
- **CORe50:** Downloaded to `core50_eval/core50_data/` with `--download` flag

---

## Authors

- Divya Sree Durga Devalla — ddeva@uic.edu
- Yuvaneswaren Ramakrishnan Sureshbabu — yrama@uic.edu
