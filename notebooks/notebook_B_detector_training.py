"""
╔══════════════════════════════════════════════════════════════╗
║  NOTEBOOK B — DETECTOR TRAINING                              ║
║  USTHB Anti-UAV Thesis                                       ║
║                                                              ║
║  Trains :                                                    ║
║    1. YOLOv8s  on DUT Anti-UAV (controlled comparison)       ║
║    2. YOLOv11s on DUT Anti-UAV (specialist detector)         ║
║    3. YOLOv11s on USTHB Anti-UAV (unified detector)          ║
║  Evaluates each model on DUT val and test splits.            ║
║  Evaluates unified model on RGBT IR and Visible val splits.  ║
║                                                              ║
║  Outputs → /kaggle/working/runs/                             ║
║                                                              ║
║  KAGGLE DATASETS TO MOUNT :                                  ║
║    - mounir2mz/dut-anti-uav                                  ║
║    - mounir2mz/usthb-anti-uav                                ║
║    - mounir2mz/antiuav-rgbt-yolo                             ║
║    (for resume) mounir2mz/model-epoch-60                     ║
╚══════════════════════════════════════════════════════════════╝
"""

import subprocess, sys
subprocess.run(["pip", "install", "ultralytics", "-q"], check=False)

from ultralytics import YOLO
from pathlib import Path
import yaml, shutil

RUNS_DIR  = Path("/kaggle/working/runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)

DUT_BASE  = Path("/kaggle/input/datasets/mounir2mz/dut-anti-uav")
USTHB_BASE= Path("/kaggle/input/datasets/mounir2mz/usthb-anti-uav/unified_yolo")
RGBT_BASE = Path("/kaggle/input/datasets/mounir2mz/antiuav-rgbt-yolo/antiuav_rgbt_yolo")
EPOCH60   = Path("/kaggle/input/datasets/mounir2mz/model-epoch-60/epoch60.pt")

# ── Helper : write data yaml ───────────────────────────────────
def write_yaml(path, train, val, nc=1, names=None):
    if names is None: names = ["uav"]
    with open(path, "w") as f:
        yaml.dump({"train": str(train), "val": str(val),
                   "nc": nc, "names": names}, f)

# ══════════════════════════════════════════════════════════════
# ── STEP 1 : Prepare DUT YAML ─────────────────────────────────
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1 — Prepare DUT YAML")
print("=" * 60)

# DUT uses YOLO format after conversion (labels/*.txt)
# If raw XML, use the conversion script from Notebook A first.
# Here we assume YOLO-format labels exist at dut-anti-uav/
DUT_YAML = Path("/kaggle/working/dut_data.yaml")
write_yaml(DUT_YAML,
           train=DUT_BASE / "train" / "train" / "img",
           val=DUT_BASE / "val" / "val" / "img")
print(f"  Written: {DUT_YAML}")

# ══════════════════════════════════════════════════════════════
# ── STEP 2 : Train YOLOv8s (controlled comparison) ───────────
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2 — Train YOLOv8s (controlled comparison)")
print("=" * 60)

model_v8 = YOLO("yolov8s.pt")
model_v8.train(
    data      = str(DUT_YAML),
    epochs    = 150,
    imgsz     = 640,
    batch     = 16,
    optimizer = "SGD",
    patience  = 20,
    mosaic    = 1.0,
    translate = 0.1,
    scale     = 0.5,
    project   = str(RUNS_DIR),
    name      = "yolov8s_specialist",
    exist_ok  = True,
    device    = 0,
    seed      = 42,
)
print("  ✅ YOLOv8s training complete")

# Evaluate on val
metrics_v8_val = model_v8.val(
    data=str(DUT_YAML), imgsz=640, conf=0.001, iou=0.70,
    device=0, verbose=True)
print(f"  YOLOv8s VAL — mAP50={metrics_v8_val.box.map50:.4f}  "
      f"mAP50-95={metrics_v8_val.box.map:.4f}  "
      f"P={metrics_v8_val.box.mp:.4f}  R={metrics_v8_val.box.mr:.4f}")

# ══════════════════════════════════════════════════════════════
# ── STEP 3 : Train YOLOv11s specialist ────────────────────────
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3 — Train YOLOv11s specialist (DUT only)")
print("=" * 60)

model_v11_spec = YOLO("yolo11s.pt")
model_v11_spec.train(
    data      = str(DUT_YAML),
    epochs    = 150,
    imgsz     = 640,
    batch     = 16,
    optimizer = "SGD",
    patience  = 20,
    mosaic    = 1.0,
    translate = 0.1,
    scale     = 0.5,
    project   = str(RUNS_DIR),
    name      = "yolov11s_specialist",
    exist_ok  = True,
    device    = 0,
    seed      = 42,
)
print("  ✅ YOLOv11s specialist training complete")

# Evaluate on val
metrics_spec_val = model_v11_spec.val(
    data=str(DUT_YAML), imgsz=640, conf=0.001, iou=0.70,
    device=0, verbose=True)
print(f"  YOLOv11s SPEC VAL — mAP50={metrics_spec_val.box.map50:.4f}  "
      f"mAP50-95={metrics_spec_val.box.map:.4f}  "
      f"P={metrics_spec_val.box.mp:.4f}  R={metrics_spec_val.box.mr:.4f}")

# Evaluate on TEST (once, final)
DUT_TEST_YAML = Path("/kaggle/working/dut_test.yaml")
write_yaml(DUT_TEST_YAML,
           train=DUT_BASE / "train" / "train" / "img",
           val=DUT_BASE / "test" / "test" / "img")
metrics_spec_test = model_v11_spec.val(
    data=str(DUT_TEST_YAML), imgsz=640, conf=0.001, iou=0.70,
    device=0, verbose=True)
print(f"  YOLOv11s SPEC TEST — mAP50={metrics_spec_test.box.map50:.4f}  "
      f"mAP50-95={metrics_spec_test.box.map:.4f}  "
      f"P={metrics_spec_test.box.mp:.4f}  R={metrics_spec_test.box.mr:.4f}")

# ══════════════════════════════════════════════════════════════
# ── STEP 4 : Train YOLOv11s unified (USTHB Anti-UAV) ─────────
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4 — Train YOLOv11s unified (USTHB Anti-UAV)")
print("=" * 60)

USTHB_YAML = Path("/kaggle/working/usthb_data.yaml")
with open(USTHB_YAML, "w") as f:
    yaml.dump({
        "path" : str(USTHB_BASE),
        "train": "images/train",
        "val"  : "images/val",
        "nc"   : 1,
        "names": ["uav"]
    }, f)
print(f"  Written: {USTHB_YAML}")

# Resume from epoch60 if available, else train from scratch
if EPOCH60.exists():
    print(f"  Resuming from {EPOCH60}")
    model_unified = YOLO(str(EPOCH60))
    model_unified.train(
        data      = str(USTHB_YAML),
        epochs    = 5,          # resume 5 more epochs
        imgsz     = 640,
        batch     = 32,
        optimizer = "SGD",
        lr0       = 0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
        warmup_epochs=3, patience=15,
        mosaic=1.0, mixup=0.1, hsv_h=0.01, hsv_s=0.3, hsv_v=0.4,
        flipud=0.5, fliplr=0.5,
        project   = str(RUNS_DIR),
        name      = "yolov11s_unified_resume",
        exist_ok  = True,
        device    = "0,1",
        seed      = 42,
    )
else:
    print("  Training from scratch (epoch60.pt not found)")
    model_unified = YOLO("yolo11s.pt")
    model_unified.train(
        data      = str(USTHB_YAML),
        epochs    = 80,
        imgsz     = 640,
        batch     = 32,
        optimizer = "SGD",
        lr0       = 0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
        warmup_epochs=3, patience=15,
        mosaic=1.0, mixup=0.1, hsv_h=0.01, hsv_s=0.3, hsv_v=0.4,
        flipud=0.5, fliplr=0.5,
        project   = str(RUNS_DIR),
        name      = "yolov11s_unified",
        exist_ok  = True,
        device    = "0,1",
        seed      = 42,
    )
print("  ✅ Unified training complete")

# ══════════════════════════════════════════════════════════════
# ── STEP 5 : Evaluate unified on all domains ──────────────────
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5 — Evaluate unified detector on all domains")
print("=" * 60)

# DUT test (held-out)
m_dut = model_unified.val(data=str(DUT_TEST_YAML), imgsz=640,
                           conf=0.001, iou=0.70, device=0, verbose=True)
print(f"  UNIFIED on DUT test — mAP50={m_dut.box.map50:.4f}  "
      f"mAP50-95={m_dut.box.map:.4f}  P={m_dut.box.mp:.4f}  R={m_dut.box.mr:.4f}")

# RGBT IR val
for modality in ["infrared", "visible"]:
    rgbt_yaml = Path(f"/kaggle/working/rgbt_{modality}.yaml")
    with open(rgbt_yaml, "w") as f:
        yaml.dump({
            "path" : str(RGBT_BASE / modality),
            "train": "images/train",
            "val"  : "images/val",
            "nc"   : 1, "names": ["uav"]}, f)
    m = model_unified.val(data=str(rgbt_yaml), imgsz=640,
                          conf=0.001, iou=0.70, device=0, verbose=True)
    print(f"  UNIFIED on RGBT {modality} val — mAP50={m.box.map50:.4f}  "
          f"mAP50-95={m.box.map:.4f}  P={m.box.mp:.4f}  R={m.box.mr:.4f}")

# Comparison table
import pandas as pd
results = pd.DataFrame([
    {"Model": "YOLOv8s specialist",  "Dataset": "DUT val",
     "mAP50": round(metrics_v8_val.box.map50, 4),
     "mAP50-95": round(metrics_v8_val.box.map, 4),
     "P": round(metrics_v8_val.box.mp, 4),
     "R": round(metrics_v8_val.box.mr, 4)},
    {"Model": "YOLOv11s specialist", "Dataset": "DUT val",
     "mAP50": round(metrics_spec_val.box.map50, 4),
     "mAP50-95": round(metrics_spec_val.box.map, 4),
     "P": round(metrics_spec_val.box.mp, 4),
     "R": round(metrics_spec_val.box.mr, 4)},
    {"Model": "YOLOv11s specialist", "Dataset": "DUT test",
     "mAP50": round(metrics_spec_test.box.map50, 4),
     "mAP50-95": round(metrics_spec_test.box.map, 4),
     "P": round(metrics_spec_test.box.mp, 4),
     "R": round(metrics_spec_test.box.mr, 4)},
    {"Model": "YOLOv11s unified", "Dataset": "DUT test",
     "mAP50": round(m_dut.box.map50, 4),
     "mAP50-95": round(m_dut.box.map, 4),
     "P": round(m_dut.box.mp, 4),
     "R": round(m_dut.box.mr, 4)},
])
results.to_csv(str(RUNS_DIR / "detection_results.csv"), index=False)
print("\n" + results.to_string(index=False))
print("\n✅  Notebook B complete → /kaggle/working/runs/")
