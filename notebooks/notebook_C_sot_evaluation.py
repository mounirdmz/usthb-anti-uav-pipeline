"""
╔══════════════════════════════════════════════════════════════╗
║  NOTEBOOK C — SOT EVALUATION                                 ║
║  USTHB Anti-UAV Thesis                                       ║
║                                                              ║
║  Benchmarks 5 trackers on DUT Anti-UAV tracking subset :    ║
║    UAVTracker, ByteTrack, OC-SORT, DeepSORT, BoT-SORT        ║
║                                                              ║
║  Metrics : SR@0.50, Mean IoU, Jitter Reduction, FPS          ║
║  + FSM state distribution per sequence (UAVTracker)          ║
║                                                              ║
║  Outputs → /kaggle/working/sot_benchmark/                    ║
║                                                              ║
║  KAGGLE DATASETS TO MOUNT :                                  ║
║    - mounir2mz/exp-v11s   (specialist weights best.pt)       ║
║    - dut-anti-uav-tracking-v0                                ║
╚══════════════════════════════════════════════════════════════╝
"""

import subprocess, sys
subprocess.run(["pip", "install", "ultralytics", "boxmot",
                "deep-sort-realtime", "-q"], check=False)

import cv2, time, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from boxmot import ByteTrack, OcSort, BoTrack

WEIGHTS      = Path("/kaggle/input/datasets/mounir2mz/exp-v11s/best.pt")
TRACKING_DIR = Path("/kaggle/input/dut-anti-uav-tracking-v0/Anti-UAV-Tracking-V0")
GT_DIR       = TRACKING_DIR   # .txt GT files in same folder
OUT_DIR      = Path("/kaggle/working/sot_benchmark")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONF_THRESH = 0.25
IMGSZ       = 640
DEVICE      = 0
WARMUP      = 10
IOU_THRESH  = 0.50   # SR threshold

sequences = sorted([d for d in TRACKING_DIR.iterdir()
                    if d.is_dir() and d.name.startswith("video")])
print(f"Sequences found : {len(sequences)}")

model = YOLO(str(WEIGHTS))
print("✅  Detector loaded\n")

# ── GT loader ─────────────────────────────────────────────────
def load_gt(seq_name):
    gt_file = GT_DIR / f"{seq_name}.txt"
    gt = {}
    if not gt_file.exists():
        return gt
    for line in gt_file.read_text().splitlines():
        line = line.strip()
        if not line: continue
        parts = line.split()
        if len(parts) < 4: continue
        try:
            fid = int(parts[0])
            x1, y1, x2, y2 = map(float, parts[1:5])
            gt[fid] = (x1, y1, x2, y2)
        except: continue
    return gt

def iou_box(a, b):
    ix1 = max(a[0],b[0]); iy1 = max(a[1],b[1])
    ix2 = min(a[2],b[2]); iy2 = min(a[3],b[3])
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    ua = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua > 0 else 0.0

# ── Alpha-Beta filter ─────────────────────────────────────────
class AlphaBeta:
    def __init__(self, alpha=0.45, beta=0.08):
        self.alpha = alpha; self.beta = beta
        self.x = self.v = 0.0
    def init(self, x): self.x = x; self.v = 0.0
    def predict(self): self.x += self.v; return self.x
    def update(self, meas):
        err = meas - self.x
        self.x += self.alpha * err
        self.v += self.beta * err
        return self.x

# ── UAVTracker ────────────────────────────────────────────────
class UAVTracker:
    TAU_CONF = 0.25; TAU_IOU = 0.30; T_MAX = 30
    def __init__(self):
        self.state = "SCAN"
        self.filters = {k: AlphaBeta() for k in ["cx","cy","w","h"]}
        self.pred_cnt = 0
        self.cx=self.cy=self.w=self.h = 0.0

    def _box(self): return (self.cx-self.w/2, self.cy-self.h/2,
                             self.cx+self.w/2, self.cy+self.h/2)

    def update(self, dets):
        for f in self.filters.values(): f.predict()
        if self.state == "SCAN":
            valid = [d for d in dets if d[4] >= self.TAU_CONF]
            if valid:
                best = max(valid, key=lambda d: d[4])
                self.cx, self.cy = (best[0]+best[2])/2, (best[1]+best[3])/2
                self.w,  self.h  = best[2]-best[0], best[3]-best[1]
                for k, v in zip(["cx","cy","w","h"],
                                 [self.cx,self.cy,self.w,self.h]):
                    self.filters[k].init(v)
                self.state = "LOCKED"; self.pred_cnt = 0
        elif self.state in ("LOCKED","PREDICT"):
            est = self._box()
            best_iou, best = 0, None
            for d in dets:
                s = iou_box(est, d[:4])
                if s > best_iou: best_iou, best = s, d
            if best_iou >= self.TAU_IOU:
                bx = (best[0]+best[2])/2; by = (best[1]+best[3])/2
                bw = best[2]-best[0];     bh = best[3]-best[1]
                self.cx = self.filters["cx"].update(bx)
                self.cy = self.filters["cy"].update(by)
                self.w  = self.filters["w"].update(bw)
                self.h  = self.filters["h"].update(bh)
                self.state = "LOCKED"; self.pred_cnt = 0
            else:
                self.pred_cnt += 1
                self.state = "PREDICT"
                if self.pred_cnt >= self.T_MAX:
                    self.state = "SCAN"; self.pred_cnt = 0
        return self._box() if self.state != "SCAN" else None, self.state

# ── Generic MOT tracker runner (returns best IoU track) ───────
def run_mot_tracker(name, factory):
    rows = []
    for seq in sequences:
        frames = sorted(seq.glob("*.jpg"))
        gt     = load_gt(seq.name)
        tracker = factory()
        t_list  = []
        iou_vals = []
        success  = 0

        for fi, fpath in enumerate(frames):
            img = cv2.imread(str(fpath))
            if img is None: continue
            t0 = time.perf_counter()
            res = model(img, imgsz=IMGSZ, conf=CONF_THRESH,
                        device=DEVICE, verbose=False)[0]
            if res.boxes is not None and len(res.boxes) > 0:
                det_arr = np.array([
                    [*b.xyxy[0].cpu().numpy(), float(b.conf[0]), 0]
                    for b in res.boxes], dtype=np.float32)
            else:
                det_arr = np.empty((0,6), dtype=np.float32)
            tracks = tracker.update(det_arr, img)
            t1 = time.perf_counter()
            if fi >= WARMUP: t_list.append((t1-t0)*1000)

            gt_box = gt.get(fi+1)
            if gt_box is None: continue
            best_iou = 0
            for t in tracks:
                s = iou_box(t[:4], gt_box)
                if s > best_iou: best_iou = s
            iou_vals.append(best_iou)
            if best_iou >= IOU_THRESH: success += 1

        sr   = success / max(len(iou_vals), 1)
        miou = float(np.mean(iou_vals)) if iou_vals else 0.0
        fps  = 1000.0 / np.mean(t_list) if t_list else 0.0
        print(f"  [{name}] {seq.name}: SR={sr:.3f}  IoU={miou:.3f}  FPS={fps:.1f}")
        rows.append({"Tracker": name, "Sequence": seq.name,
                     "SR": round(sr,4), "IoU_mean": round(miou,4),
                     "FPS": round(fps,1)})
    return rows

# ── UAVTracker runner ─────────────────────────────────────────
def run_uavtracker():
    rows = []
    for seq in sequences:
        frames = sorted(seq.glob("*.jpg"))
        gt = load_gt(seq.name)
        tracker = UAVTracker()
        t_list = []; iou_vals = []
        state_counts = {"LOCKED":0,"PREDICT":0,"SCAN":0}
        raw_cx = []; filt_cx = []

        for fi, fpath in enumerate(frames):
            img = cv2.imread(str(fpath))
            if img is None: continue
            t0 = time.perf_counter()
            res = model(img, imgsz=IMGSZ, conf=CONF_THRESH,
                        device=DEVICE, verbose=False)[0]
            dets = []
            if res.boxes is not None:
                for b in res.boxes:
                    x1,y1,x2,y2 = b.xyxy[0].cpu().numpy()
                    dets.append((x1,y1,x2,y2,float(b.conf[0])))
                    if fi == 0 or len(raw_cx) == fi:
                        raw_cx.append((x1+x2)/2)
            box, state = tracker.update(dets)
            t1 = time.perf_counter()
            if fi >= WARMUP: t_list.append((t1-t0)*1000)
            state_counts[state] = state_counts.get(state, 0) + 1
            if box: filt_cx.append((box[0]+box[2])/2)

            gt_box = gt.get(fi+1)
            if gt_box is None: continue
            iou_v = iou_box(box, gt_box) if box else 0.0
            iou_vals.append(iou_v)

        sr   = sum(1 for v in iou_vals if v >= IOU_THRESH) / max(len(iou_vals),1)
        miou = float(np.mean(iou_vals)) if iou_vals else 0.0
        fps  = 1000.0 / np.mean(t_list) if t_list else 0.0
        total = sum(state_counts.values())
        locked_pct  = 100*state_counts["LOCKED"]/max(total,1)
        predict_pct = 100*state_counts["PREDICT"]/max(total,1)
        scan_pct    = 100*state_counts["SCAN"]/max(total,1)

        # Jitter reduction
        rc = np.array(raw_cx[:len(filt_cx)])
        fc = np.array(filt_cx)
        if len(rc) > 1 and len(fc) > 1:
            raw_jitter  = np.var(np.diff(rc))
            filt_jitter = np.var(np.diff(fc))
            jr = (raw_jitter - filt_jitter) / max(raw_jitter, 1e-6) * 100
        else:
            jr = 0.0

        print(f"  [UAVTracker] {seq.name}: SR={sr:.3f}  IoU={miou:.3f}  "
              f"JR={jr:.1f}%  FPS={fps:.1f}  "
              f"L={locked_pct:.1f}% P={predict_pct:.1f}% S={scan_pct:.1f}%")
        rows.append({"Tracker": "UAVTracker", "Sequence": seq.name,
                     "SR": round(sr,4), "IoU_mean": round(miou,4),
                     "JR_pct": round(jr,1), "FPS": round(fps,1),
                     "LOCKED_pct": round(locked_pct,1),
                     "PREDICT_pct": round(predict_pct,1),
                     "SCAN_pct": round(scan_pct,1)})
    return rows

# ── Run all trackers ───────────────────────────────────────────
print("\n[1/5] UAVTracker")
uav_rows = run_uavtracker()

print("\n[2/5] ByteTrack")
bt_rows  = run_mot_tracker("ByteTrack", lambda: ByteTrack())

print("\n[3/5] OC-SORT")
oc_rows  = run_mot_tracker("OC-SORT", lambda: OcSort())

print("\n[4/5] DeepSORT")
from deep_sort_realtime.deepsort_tracker import DeepSort
def ds_factory():
    return DeepSort(max_age=30, n_init=1, max_iou_distance=0.70,
                    max_cosine_distance=0.4, embedder="mobilenet",
                    embedder_gpu=True)

def run_deepsort():
    rows = []
    for seq in sequences:
        frames = sorted(seq.glob("*.jpg"))
        gt = load_gt(seq.name)
        tracker = ds_factory()
        t_list = []; iou_vals = []
        for fi, fpath in enumerate(frames):
            img = cv2.imread(str(fpath))
            if img is None: continue
            t0 = time.perf_counter()
            res = model(img, imgsz=IMGSZ, conf=CONF_THRESH,
                        device=DEVICE, verbose=False)[0]
            ds_dets = []
            if res.boxes is not None:
                for b in res.boxes:
                    x1,y1,x2,y2 = b.xyxy[0].cpu().numpy()
                    ds_dets.append(([x1,y1,x2-x1,y2-y1], float(b.conf[0]), 0))
            tracks = tracker.update_tracks(ds_dets, frame=img)
            conf = [t for t in tracks if t.is_confirmed()]
            t1 = time.perf_counter()
            if fi >= WARMUP: t_list.append((t1-t0)*1000)
            gt_box = gt.get(fi+1)
            if gt_box is None: continue
            best_iou = 0
            for t in conf:
                s = iou_box(t.to_ltrb(), gt_box)
                if s > best_iou: best_iou = s
            iou_vals.append(best_iou)
        sr   = sum(1 for v in iou_vals if v>=IOU_THRESH)/max(len(iou_vals),1)
        miou = float(np.mean(iou_vals)) if iou_vals else 0.0
        fps  = 1000.0/np.mean(t_list) if t_list else 0.0
        print(f"  [DeepSORT] {seq.name}: SR={sr:.3f}  IoU={miou:.3f}  FPS={fps:.1f}")
        rows.append({"Tracker":"DeepSORT","Sequence":seq.name,
                     "SR":round(sr,4),"IoU_mean":round(miou,4),"FPS":round(fps,1)})
    return rows

ds_rows = run_deepsort()

print("\n[5/5] BoT-SORT")
bot_rows = run_mot_tracker("BoT-SORT", lambda: BoTrack())

# ── Aggregate and save ─────────────────────────────────────────
all_rows = uav_rows + bt_rows + oc_rows + ds_rows + bot_rows
df = pd.DataFrame(all_rows)
df.to_csv(OUT_DIR / "sot_results_per_sequence.csv", index=False)

# Summary
summary_rows = []
for name in ["UAVTracker","ByteTrack","OC-SORT","DeepSORT","BoT-SORT"]:
    sub = df[df["Tracker"] == name]
    summary_rows.append({
        "Tracker": name,
        "SR_mean": round(sub["SR"].mean(), 4),
        "IoU_mean": round(sub["IoU_mean"].mean(), 4),
        "FPS_mean": round(sub["FPS"].mean(), 1),
    })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR / "sot_results_summary.csv", index=False)

print("\n" + "="*60)
print("  SOT BENCHMARK SUMMARY")
print("="*60)
print(summary_df.to_string(index=False))

# Figure — comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("SOT Benchmark — DUT Anti-UAV (20 sequences)", fontsize=13)
colors = ["#2196F3","#4CAF50","#FF9800","#F44336","#9C27B0"]
for ax, metric, title in [
    (axes[0], "SR_mean",  "Success Rate (SR@0.50)"),
    (axes[1], "IoU_mean", "Mean IoU"),
    (axes[2], "FPS_mean", "FPS"),
]:
    bars = ax.bar(summary_df["Tracker"], summary_df[metric], color=colors)
    ax.set_title(title); ax.set_ylabel(metric)
    for bar, val in zip(bars, summary_df[metric]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, summary_df[metric].max() * 1.15)
plt.tight_layout()
plt.savefig(OUT_DIR / "sot_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\n✅  Notebook C complete → {OUT_DIR}/")
