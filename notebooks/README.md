# USTHB Anti-UAV — Experimental Pipeline
## Complete Notebooks for Reproducibility

This folder contains the 5 functional notebooks (implemented as .py scripts
for Kaggle) that constitute the complete experimental pipeline of the thesis.

---

## Overview

| Notebook | File | Purpose | GPU | Est. Time |
|---|---|---|---|---|
| A | notebook_A_dataset_audit.py | Dataset statistics & figures | No | ~15 min |
| B | notebook_B_detector_training.py | YOLOv11s training | T4×2 | ~5-7 hrs |
| C | notebook_C_sot_evaluation.py | SOT benchmark (5 trackers) | T4×2 | ~2 hrs |
| D | notebook_D_mot_evaluation.py | MOT benchmark (ByteTrack, OC-SORT) | T4×2 | ~3 hrs |
| E | notebook_E_figures_generation.py | All thesis figures | T4×2 | ~30 min |

---

## Execution Order

Notebooks must be executed in order — each depends on the outputs of the previous:

```
A → B → C → D → E
```

B produces the weights used by C, D, E.
C produces tracking_results.csv used by E.
D produces mot_results_summary.csv used by E.

---

## Kaggle Setup (for each notebook)

### Datasets to mount (Add Data → search by name):
- `mounir2mz/dut-anti-uav`           → DUT Anti-UAV (detection + tracking)
- `mounir2mz/antiuav-rgbt-yolo`      → Anti-UAV RGBT (IR + visible)
- `mounir2mz/multiuav-yolo`          → MultiUAV_Train (YOLO format)
- `mounir2mz/usthb-anti-uav`         → Unified dataset (Notebook B onward)
- `mounir2mz/exp-v11s`               → Specialist weights (Notebook C, E)
- `mounir2mz/unified-detector`       → Unified weights (Notebooks C, D, E)
- `dut-anti-uav-tracking-v0`         → DUT tracking sequences

### Settings:
- Accelerator: GPU T4×2
- Internet: ON (for pip installs)
- Persistence: ON (to save outputs between sessions)

---

## Key Results Summary

### Detection (YOLOv11s)
| Model | Dataset | mAP@0.50 | mAP@0.50:95 | P | R |
|---|---|---|---|---|---|
| Specialist | DUT test | 0.9612 | 0.6900 | 0.9744 | 0.9296 |
| Unified | DUT test | 0.9177 | 0.6368 | 0.9210 | 0.8563 |
| Unified | RGBT IR val | 0.9929 | 0.6424 | 0.9915 | 0.9879 |
| Unified | RGBT Vis val | 0.9853 | 0.6753 | 0.9621 | 0.9792 |

### SOT Benchmark (DUT tracking, 20 seqs, 24 804 frames)
| Tracker | SR | IoU | FPS |
|---|---|---|---|
| UAVTracker | 0.9062 | 0.7489 | 97.0 |
| ByteTrack | 0.9597 | 0.7946 | 80.6 |
| OC-SORT | 0.9486 | 0.8000 | 84.9 |
| DeepSORT | 0.9001 | 0.7517 | 34.4 |
| BoT-SORT | 0.9066 | 0.7132 | 21.2 |

### MOT Benchmark (MultiUAV val, 40 seqs, ~30 000 frames)
| Tracker | MOTA | IDF1 | IDs | FP | FN | FPS |
|---|---|---|---|---|---|---|
| ByteTrack | 0.7537 | 0.5358 | 4184 | 49677 | 80149 | 63.9 |
| OC-SORT | 0.7561 | 0.5188 | 5045 | 50186 | 74281 | 63.4 |

---

## Reproducibility

- Random seed: `seed=42` everywhere
- MultiUAV val split: `random.seed(42)`, `val_ratio=0.20`, sequence-level
- All datasets published on Kaggle under `mounir2mz`
- Unified training resumed from `epoch60.pt` (session timeout at epoch 61)

---

## Environment

- Platform: Kaggle Notebooks
- OS: Ubuntu 20.04 LTS
- Python: 3.12
- GPU: NVIDIA Tesla T4×2
- Key libraries: ultralytics 8.3.x, boxmot, motmetrics 1.2.0, opencv 4.x

---

*USTHB Anti-UAV — Master Thesis*
*Lead Thesis Coding Engineer*
