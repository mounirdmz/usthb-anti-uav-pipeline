"""
╔══════════════════════════════════════════════════════════════╗
║  NOTEBOOK D — MOT EVALUATION                                 ║
║  USTHB Anti-UAV Thesis                                       ║
║                                                              ║
║  Evaluates ByteTrack and OC-SORT on MultiUAV_Train           ║
║  val split (40 sequences, ~30 000 consecutive frames)        ║
║  using full MOT metrics: MOTA, IDF1, ID switches, FP, FN    ║
║                                                              ║
║  Platform   : Kaggle Notebooks (T4×2 GPU)                   ║
║  Author     : Lead Thesis Coding Engineer                    ║
║  Dataset    : mounir2mz/multiuav-yolo (val split)           ║
║  Weights    : mounir2mz/unified-detector                     ║
╚══════════════════════════════════════════════════════════════╝

INPUTS (mount on Kaggle before running):
  /kaggle/input/datasets/mounir2mz/unified-detector/
      usthb_yolov11s_unified.pt
  /kaggle/working/multiuav_extracted/MultiUAV_Train/
      TrainVideos/   ← .mp4 files
      TrainLabels/   ← .txt files (MOT format)

  NOTE: Run download_multiuav_train.py first if videos not extracted.

OUTPUTS:
  /kaggle/working/benchmark_mot_full/
      mot_results_summary.csv      ← aggregated per tracker
      mot_results_per_sequence.csv ← per sequence breakdown
"""

# ── CELL 1 — Install dependencies ─────────────────────────────
import subprocess, sys
subprocess.run(["pip", "install",
                "motmetrics==1.2.0",
                "boxmot",
                "deep-sort-realtime",
                "ultralytics", "-q"], check=False)

# ── CELL 2 — Compatibility patches ────────────────────────────
import numpy as np
import pandas as pd

if not hasattr(np, 'asfarray'):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(pd.DataFrame, 'append'):
    pd.DataFrame.append = lambda self, other, **kw: \
        pd.concat([self, other],
                  ignore_index=kw.get('ignore_index', False))
print("✅  Compatibility patches applied")

# ── CELL 3 — Config ───────────────────────────────────────────
import cv2, time, random
import motmetrics as mm
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

WEIGHTS_PATH = Path("/kaggle/input/datasets/mounir2mz/unified-detector"
                    "/usthb_yolov11s_unified.pt")
EXTRACT_ROOT = Path("/kaggle/working/multiuav_extracted/MultiUAV_Train")
VIDEO_DIR    = EXTRACT_ROOT / "TrainVideos"
LABEL_DIR    = EXTRACT_ROOT / "TrainLabels"
OUT_DIR      = Path("/kaggle/working/benchmark_mot_full")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMGSZ         = 640
CONF_THRESH   = 0.25
DEVICE        = 0
WARMUP_FRAMES = 10
RANDOM_SEED   = 42
VAL_RATIO     = 0.20

# ── CELL 4 — Reconstruct val split (seed=42, 80/20) ───────────
video_files = sorted(VIDEO_DIR.glob("*.mp4"))
sequences   = [(vf, LABEL_DIR / (vf.stem + ".txt"))
               for vf in video_files
               if (LABEL_DIR / (vf.stem + ".txt")).exists()]

random.seed(RANDOM_SEED)
shuffled = sequences.copy()
random.shuffle(shuffled)
n_val    = max(1, round(len(shuffled) * VAL_RATIO))
val_seqs = shuffled[:n_val]

print(f"Total sequences : {len(sequences)}")
print(f"Val  sequences  : {len(val_seqs)}")
print(f"Val names       : {[v[0].stem for v in val_seqs[:5]]} ...")

total_frames = 0
for vf, _ in val_seqs:
    cap = cv2.VideoCapture(str(vf))
    total_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
print(f"Total frames    : {total_frames:,}")

# ── CELL 5 — Helper functions ─────────────────────────────────
def parse_mot_gt(label_path):
    """
    Parse MOT-format label file.
    Returns {frame_id (1-based): [(track_id, x, y, w, h), ...]}
    Only active annotations (conf==1, w>0, h>0).
    """
    gt = defaultdict(list)
    for line in Path(label_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        try:
            fid  = int(float(parts[0]))
            tid  = int(float(parts[1]))
            x,y,w,h = float(parts[2]),float(parts[3]),\
                      float(parts[4]),float(parts[5])
            conf = int(float(parts[6])) if len(parts) > 6 else 1
            if conf == 1 and w > 0 and h > 0:
                gt[fid].append((tid, x, y, w, h))
        except (ValueError, IndexError):
            continue
    return gt

def xyxy_to_xywh(x1,y1,x2,y2):
    return [x1, y1, x2-x1, y2-y1]

def seq_summary(acc):
    mh = mm.metrics.create()
    return mh.compute(acc, metrics=[
        "mota","idf1","num_switches",
        "num_false_positives","num_misses"], name="s")

def global_summary(rows):
    df = pd.DataFrame(rows)
    return {
        "MOTA"        : round(float(df["MOTA"].mean()), 4),
        "IDF1"        : round(float(df["IDF1"].mean()), 4),
        "ID_switches" : int(df["ID_switches"].sum()),
        "FP"          : int(df["FP"].sum()),
        "FN"          : int(df["FN"].sum()),
    }

# ── CELL 6 — Load detector ────────────────────────────────────
print("Loading unified detector ...")
detector = YOLO(str(WEIGHTS_PATH))
print("✅  Detector loaded\n")

# ── CELL 7 — Generic MOT tracker runner ───────────────────────
def run_mot_tracker(tracker_name, tracker_factory):
    fps_list = []
    rows     = []

    for si, (video_path, label_path) in enumerate(val_seqs):
        seq_name = video_path.stem
        cap      = cv2.VideoCapture(str(video_path))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  [{tracker_name}] {si+1}/{len(val_seqs)} "
              f"{seq_name} ({n_frames}f)", end="  ", flush=True)

        gt_dict  = parse_mot_gt(label_path)
        tracker  = tracker_factory()
        acc      = mm.MOTAccumulator(auto_id=True)
        t_list   = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            tt0 = time.perf_counter()
            result = detector(frame, imgsz=IMGSZ, conf=CONF_THRESH,
                              device=DEVICE, verbose=False)[0]
            if result.boxes is not None and len(result.boxes) > 0:
                det_arr = np.array([
                    [*b.xyxy[0].cpu().numpy(), float(b.conf[0]), 0]
                    for b in result.boxes], dtype=np.float32)
            else:
                det_arr = np.empty((0,6), dtype=np.float32)

            tracks = tracker.update(det_arr, frame)
            tt1 = time.perf_counter()
            if frame_idx > WARMUP_FRAMES:
                t_list.append((tt1-tt0)*1000)

            pred_ids  = [int(t[4]) for t in tracks]
            pred_dets = [xyxy_to_xywh(*t[:4]) for t in tracks]

            gt_frame = gt_dict.get(frame_idx, [])
            gt_ids   = [g[0] for g in gt_frame]
            gt_dets  = [[g[1],g[2],g[3],g[4]] for g in gt_frame]

            dist = mm.distances.iou_matrix(
                gt_dets, pred_dets, max_iou=0.5)
            acc.update(gt_ids, pred_ids, dist)

        cap.release()
        fps = 1000.0 / np.mean(t_list) if t_list else 0.0
        fps_list.append(fps)

        s    = seq_summary(acc)
        mota = float(s["mota"].iloc[0])
        idf1 = float(s["idf1"].iloc[0])
        ids  = int(s["num_switches"].iloc[0])
        fp   = int(s["num_false_positives"].iloc[0])
        fn   = int(s["num_misses"].iloc[0])
        print(f"→ MOTA={mota:.3f}  IDF1={idf1:.3f}  "
              f"IDs={ids}  FP={fp}  FN={fn}  FPS={fps:.1f}")

        rows.append({"Sequence": seq_name, "Tracker": tracker_name,
                     "MOTA": round(mota,4), "IDF1": round(idf1,4),
                     "ID_switches": ids, "FP": fp, "FN": fn,
                     "FPS": round(fps,1)})

    return global_summary(rows), float(np.mean(fps_list)), rows

# ── CELL 8 — Run ByteTrack ────────────────────────────────────
from boxmot import ByteTrack, OcSort

print("[1/2] ByteTrack")
bt_s, bt_fps, bt_rows = run_mot_tracker("ByteTrack", lambda: ByteTrack())
print(f"  ✅  done (mean FPS={bt_fps:.1f})\n")

# ── CELL 9 — Run OC-SORT ──────────────────────────────────────
print("[2/2] OC-SORT")
oc_s, oc_fps, oc_rows = run_mot_tracker("OC-SORT", lambda: OcSort())
print(f"  ✅  done (mean FPS={oc_fps:.1f})\n")

# ── CELL 10 — Summary ─────────────────────────────────────────
print("\n" + "=" * 70)
print("  MOT BENCHMARK — MultiUAV Val Split (full consecutive frames)")
print("  Unified YOLOv11s detector | Real MOT GT track IDs")
print("=" * 70)
print(f"\n  {'Tracker':<12} {'MOTA':>8} {'IDF1':>8} "
      f"{'IDs':>8} {'FP':>8} {'FN':>8} {'FPS':>8}")
print("  " + "-" * 64)

summary_rows = []
for name, s, fps in [("ByteTrack", bt_s, bt_fps),
                      ("OC-SORT",   oc_s, oc_fps)]:
    print(f"  {name:<12} {s['MOTA']:>8.4f} {s['IDF1']:>8.4f} "
          f"{s['ID_switches']:>8} {s['FP']:>8} {s['FN']:>8} {fps:>8.1f}")
    summary_rows.append({**s, "Tracker": name, "FPS": round(fps,1)})
print("=" * 70)

# Context: total GT instances
total_gt = sum(bt_s['FN'] + bt_s['FP'] +
               sum(r['ID_switches'] for r in bt_rows))
print(f"\n  Note: FP/FN contextualization:")
print(f"  ByteTrack false alarm rate : {100*bt_s['FP']/(bt_s['FP']+bt_s['FN']+1):.1f}%"
      f" of total detections")
print(f"  ByteTrack miss rate        : {100*bt_s['FN']/(bt_s['FP']+bt_s['FN']+1):.1f}%"
      f" of total GT instances")

all_rows = bt_rows + oc_rows
pd.DataFrame(all_rows).to_csv(
    OUT_DIR / "mot_results_per_sequence.csv", index=False)
pd.DataFrame(summary_rows).to_csv(
    OUT_DIR / "mot_results_summary.csv", index=False)

print(f"\n  Saved → {OUT_DIR}/mot_results_summary.csv")
print(f"  Saved → {OUT_DIR}/mot_results_per_sequence.csv")
print("\n✅  MOT benchmark complete.\n")
