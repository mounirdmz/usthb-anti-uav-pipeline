"""
╔══════════════════════════════════════════════════════════════╗
║  NOTEBOOK E — FIGURES GENERATION                             ║
║  USTHB Anti-UAV Thesis                                       ║
║                                                              ║
║  Generates ALL thesis figures without re-running             ║
║  experiments. Reads from CSV outputs of Notebooks A-D.      ║
║                                                              ║
║  Platform   : Kaggle Notebooks (T4×2 GPU)                   ║
║  Author     : Lead Thesis Coding Engineer                    ║
╚══════════════════════════════════════════════════════════════╝

INPUTS:
  Notebooks A-D outputs + mounted datasets + weights

OUTPUTS (all saved to /kaggle/working/figures/):
  ── Detection ──────────────────────────────────────
  confusion_matrix_specialist.png   (Fig 4.2 — Ch4)
  pr_curve_specialist.png           (Fig 4.3 — Ch4)
  training_curves_specialist.png    (Fig 4.4 — Ch4)
  detection_examples.png            (Fig 4.5 — Ch4)
  ── SOT Tracking ───────────────────────────────────
  jitter_alphabeta.png              (Fig 4.6 — Ch4)
  tracker_states.png                (Fig 4.7 — Ch4)
  ── MOT Tracking ───────────────────────────────────
  mot_examples.png                  (Fig 4.8 — Ch4)
  ── Dataset Audit ──────────────────────────────────
  (DUT figures already generated in Notebook A)
  audit_multiuav_bbox.png
  audit_multiuav_density.png
  audit_rgbt_bbox.png
  audit_rgbt_visibility.png
"""

# ── CELL 1 — Install & imports ─────────────────────────────────
import subprocess, sys
subprocess.run(["pip", "install", "ultralytics", "boxmot", "-q"],
               check=False)

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml, shutil
from pathlib import Path
from ultralytics import YOLO

FIGURES_DIR  = Path("/kaggle/working/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DPI = 150
print(f"Figures will be saved to: {FIGURES_DIR}")

# ── CELL 2 — Paths ────────────────────────────────────────────
WEIGHTS_SPEC = Path("/kaggle/input/datasets/mounir2mz/exp-v11s/best.pt")
WEIGHTS_UNIF = Path("/kaggle/input/datasets/mounir2mz/unified-detector"
                    "/usthb_yolov11s_unified.pt")
DUT_BASE     = Path("/kaggle/input/datasets/mounir2mz/dut-anti-uav")
TRACKING_DIR = Path("/kaggle/input/dut-anti-uav-tracking-v0"
                    "/Anti-UAV-Tracking-V0")
RUNS_DIR     = Path("/kaggle/working/runs")
TRACKING_CSV = Path("/kaggle/working/phase3/tracking_results.csv")
MUAV_IMG_DIR = Path("/kaggle/input/datasets/mounir2mz/multiuav-yolo"
                    "/multiuav_yolo/images/val")

# ══════════════════════════════════════════════════════════════
# FIGURE 1 — Confusion matrix + PR curve (specialist, test)
# ══════════════════════════════════════════════════════════════
print("\n[1/7] Running .val() to generate confusion matrix & PR curve ...")
try:
    model_spec = YOLO(str(WEIGHTS_SPEC))

    TEST_YAML = FIGURES_DIR / "data_test.yaml"
    yaml.dump({
        "path" : str(DUT_BASE),
        "train": "train/train/img",
        "val"  : "test/test/img",
        "nc"   : 1,
        "names": ["uav"]
    }, open(TEST_YAML, "w"))

    metrics = model_spec.val(
        data=str(TEST_YAML), imgsz=640, conf=0.001, iou=0.70,
        device=0, verbose=False,
        project=str(FIGURES_DIR / "val_runs"),
        name="specialist_test", exist_ok=True, plots=True)

    val_out = FIGURES_DIR / "val_runs" / "specialist_test"
    for src, dst in [
        ("confusion_matrix_normalized.png", "confusion_matrix_specialist.png"),
        ("PR_curve.png",                    "pr_curve_specialist.png"),
    ]:
        p = val_out / src
        if p.exists():
            shutil.copy(p, FIGURES_DIR / dst)
            print(f"  ✅  {dst}")
        else:
            print(f"  ⚠️  {src} not found")
except Exception as e:
    print(f"  ❌  {e}")

# ══════════════════════════════════════════════════════════════
# FIGURE 2 — Training curves (specialist YOLOv11s)
# ══════════════════════════════════════════════════════════════
print("\n[2/7] Training curves ...")
try:
    csv_files = list(RUNS_DIR.rglob("results.csv"))
    spec_csv  = next((f for f in csv_files
                      if any(k in str(f) for k in
                             ["yolo11s_baseline","specialist","v11s_baseline"])),
                     csv_files[0] if csv_files else None)

    if spec_csv:
        df = pd.read_csv(spec_csv)
        df.columns = df.columns.str.strip()

        COLS = {
            "train/box_loss"      : ("Box Loss — train", "#2196F3"),
            "val/box_loss"        : ("Box Loss — val",   "#FF9800"),
            "train/cls_loss"      : ("Cls Loss — train", "#2196F3"),
            "val/cls_loss"        : ("Cls Loss — val",   "#FF9800"),
            "metrics/mAP50(B)"    : ("mAP@0.50",         "#4CAF50"),
            "metrics/mAP50-95(B)" : ("mAP@0.50:0.95",    "#9C27B0"),
        }
        avail = {k: v for k, v in COLS.items() if k in df.columns}

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle("Courbes d'entraînement — Détecteur spécialiste "
                     "YOLOv11s (DUT Anti-UAV)", fontsize=13,
                     fontweight="bold")

        for ax, (col, (label, color)) in zip(axes.flat, avail.items()):
            ax.plot(df["epoch"], df[col], color=color, linewidth=1.5)
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("Époque"); ax.grid(True, alpha=0.3)

        for ax in axes.flat[len(avail):]:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "training_curves_specialist.png",
                    dpi=DPI, bbox_inches="tight")
        plt.close()
        print(f"  ✅  training_curves_specialist.png (from {spec_csv.name})")
    else:
        print("  ⚠️  No results.csv found")
except Exception as e:
    print(f"  ❌  {e}")

# ══════════════════════════════════════════════════════════════
# FIGURE 3 — Detection examples grid
# ══════════════════════════════════════════════════════════════
print("\n[3/7] Detection examples ...")
try:
    model_spec = YOLO(str(WEIGHTS_SPEC))
    test_imgs  = sorted((DUT_BASE / "test/test/img").glob("*.jpg"))
    step       = max(1, len(test_imgs) // 8)
    selected   = [test_imgs[i * step] for i in range(min(8, len(test_imgs)))]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Exemples de détections — Détecteur spécialiste "
                 "YOLOv11s (partition de test DUT Anti-UAV)",
                 fontsize=12, fontweight="bold")

    for ax, img_path in zip(axes.flat, selected):
        img     = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res     = model_spec(img, imgsz=640, conf=0.25,
                             device=0, verbose=False)[0]
        for box in (res.boxes or []):
            x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0])
            cv2.rectangle(img_rgb,(x1,y1),(x2,y2),(255,50,50),2)
            cv2.putText(img_rgb, f"{conf:.2f}", (x1, max(y1-5,12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,50,50), 2)
        ax.imshow(img_rgb); ax.axis("off")
        ax.set_title(img_path.stem[:20], fontsize=7)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "detection_examples.png",
                dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  ✅  detection_examples.png")
except Exception as e:
    print(f"  ❌  {e}")

# ══════════════════════════════════════════════════════════════
# FIGURE 4 — Jitter: raw vs Alpha-Beta (5 sequences)
# ══════════════════════════════════════════════════════════════
print("\n[4/7] Jitter Alpha-Beta ...")
try:
    ALPHA, BETA = 0.45, 0.08
    TARGET_SEQS = {
        "video02": "JR = 72.6%",
        "video03": "JR = 86.3%",
        "video09": "JR = 40.1%",
        "video14": "JR = 33.3%",
        "video15": "JR = −7.7%",
    }

    def alphabeta(raw):
        x_e, v_e = raw[0], 0.0
        out = []
        for x in raw:
            xp = x_e + v_e
            x_e = xp + ALPHA*(x - xp)
            v_e = v_e + BETA*(x - xp)
            out.append(x_e)
        return np.array(out)

    model_spec = YOLO(str(WEIGHTS_SPEC))
    fig, axes  = plt.subplots(len(TARGET_SEQS), 1,
                              figsize=(14, 3*len(TARGET_SEQS)))
    fig.suptitle("Coordonnées brutes vs filtrées Alpha-Bêta "
                 "(α=0.45, β=0.08)", fontsize=13, fontweight="bold")

    for ax, (seq, label) in zip(axes, TARGET_SEQS.items()):
        seq_dir = TRACKING_DIR / seq
        if not seq_dir.exists():
            ax.set_title(f"{seq} — introuvable"); continue

        cx_raw = []
        for fp in sorted(seq_dir.glob("*.jpg")):
            img = cv2.imread(str(fp))
            if img is None: continue
            res = model_spec(img, imgsz=640, conf=0.25,
                             device=0, verbose=False)[0]
            if res.boxes and len(res.boxes) > 0:
                b = res.boxes[0]
                x1,y1,x2,y2 = b.xyxy[0].cpu().numpy()
                cx_raw.append((x1+x2)/2)
            else:
                cx_raw.append(np.nan)

        cx_raw = np.array(cx_raw)
        valid  = ~np.isnan(cx_raw)
        cx_flt = cx_raw.copy()
        if valid.sum() > 1:
            cx_flt[valid] = alphabeta(cx_raw[valid])

        x = np.arange(len(cx_raw))
        ax.plot(x, cx_raw, color="#e74c3c", lw=0.8,
                alpha=0.7, label="Brut (YOLOv11s)")
        ax.plot(x, cx_flt, color="#2980b9", lw=1.5,
                label="Filtré Alpha-Bêta")
        ax.set_title(f"{seq}  |  {label}", fontsize=10)
        ax.set_xlabel("Frame"); ax.set_ylabel("cx (px)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "jitter_alphabeta.png",
                dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  ✅  jitter_alphabeta.png")
except Exception as e:
    print(f"  ❌  {e}")

# ══════════════════════════════════════════════════════════════
# FIGURE 5 — Tracker states (LOCKED / PREDICT / SCAN) video17
# ══════════════════════════════════════════════════════════════
print("\n[5/7] Tracker states (video17) ...")
try:
    ALPHA,BETA   = 0.45, 0.08
    TAU_C,TAU_I  = 0.25, 0.30
    T_MAX        = 30

    def iou_xywh(a, b):
        ax1=a[0]-a[2]/2; ay1=a[1]-a[3]/2
        ax2=a[0]+a[2]/2; ay2=a[1]+a[3]/2
        bx1=b[0]-b[2]/2; by1=b[1]-b[3]/2
        bx2=b[0]+b[2]/2; by2=b[1]+b[3]/2
        iw=max(0,min(ax2,bx2)-max(ax1,bx1))
        ih=max(0,min(ay2,by2)-max(ay1,by1))
        inter=iw*ih
        union=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
        return inter/union if union>0 else 0

    model_spec = YOLO(str(WEIGHTS_SPEC))
    seq_dir    = TRACKING_DIR / "video17"
    frames     = sorted(seq_dir.glob("*.jpg"))

    state       = "SCAN"
    xe=ye=we=he = 0.0
    vx=vy=vw=vh = 0.0
    pred_cnt    = 0
    captured    = {"LOCKED": None, "PREDICT": None, "SCAN": None}

    for fp in frames:
        img = cv2.imread(str(fp))
        if img is None: continue
        res  = model_spec(img, imgsz=640, conf=0.001,
                          device=0, verbose=False)[0]
        dets = []
        if res.boxes:
            for b in res.boxes:
                if float(b.conf[0]) >= TAU_C:
                    x1,y1,x2,y2 = b.xyxy[0].cpu().numpy()
                    dets.append(((x1+x2)/2,(y1+y2)/2,
                                 x2-x1,y2-y1,float(b.conf[0])))

        if state == "SCAN":
            if dets:
                best = max(dets, key=lambda d: d[4])
                xe,ye,we,he = best[:4]
                vx=vy=vw=vh=0.0; pred_cnt=0; state="LOCKED"

        elif state == "LOCKED":
            xp=xe+vx; yp=ye+vy; wp=we+vw; hp=he+vh
            bi,bd = 0, None
            for d in dets:
                s=iou_xywh((xp,yp,wp,hp),(d[0],d[1],d[2],d[3]))
                if s>bi: bi,bd=s,d
            if bi>=TAU_I and bd:
                xe=xp+ALPHA*(bd[0]-xp); ye=yp+ALPHA*(bd[1]-yp)
                we=wp+ALPHA*(bd[2]-wp); he=hp+ALPHA*(bd[3]-hp)
                vx+=BETA*(bd[0]-xp); vy+=BETA*(bd[1]-yp)
                vw+=BETA*(bd[2]-wp); vh+=BETA*(bd[3]-hp)
            else:
                pred_cnt=1; state="PREDICT"

        elif state == "PREDICT":
            xe+=vx; ye+=vy; we+=vw; he+=vh
            bi,bd = 0, None
            for d in dets:
                s=iou_xywh((xe,ye,we,he),(d[0],d[1],d[2],d[3]))
                if s>bi: bi,bd=s,d
            if bi>=TAU_I and bd:
                xe=xe+ALPHA*(bd[0]-xe); ye=ye+ALPHA*(bd[1]-ye)
                pred_cnt=0; state="LOCKED"
            else:
                pred_cnt+=1
                if pred_cnt>=T_MAX: state="SCAN"; pred_cnt=0

        if captured.get(state) is None:
            captured[state] = (img.copy(), xe, ye, we, he)
        if all(v is not None for v in captured.values()):
            break

    COLORS = {"LOCKED":(0,200,0), "PREDICT":(255,165,0), "SCAN":(200,0,0)}
    TITLES = {
        "LOCKED" : "(a) LOCKED — suivi nominal",
        "PREDICT": "(b) PREDICT — coasting",
        "SCAN"   : "(c) SCAN — recherche",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("États de la FSM UAVTracker — séquence video17",
                 fontsize=13, fontweight="bold")

    for ax, s in zip(axes, ["LOCKED","PREDICT","SCAN"]):
        data = captured[s]
        if data is None:
            ax.set_title(f"{s} — non atteint"); continue
        img, x_e, y_e, w_e, h_e = data
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        col = COLORS[s]
        x1=int(x_e-w_e/2); y1=int(y_e-h_e/2)
        x2=int(x_e+w_e/2); y2=int(y_e+h_e/2)
        cv2.rectangle(rgb,(x1,y1),(x2,y2),col,2)
        cv2.putText(rgb, s, (x1, max(y1-8,12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        ax.imshow(rgb); ax.axis("off")
        ax.set_title(TITLES[s], fontsize=10)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tracker_states.png",
                dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  ✅  tracker_states.png")
except Exception as e:
    print(f"  ❌  {e}")

# ══════════════════════════════════════════════════════════════
# FIGURE 6 — MOT examples: ByteTrack vs OC-SORT (best sequence)
# ══════════════════════════════════════════════════════════════
print("\n[6/7] MOT examples (ByteTrack vs OC-SORT) ...")
try:
    from boxmot import ByteTrack, OcSort

    model_unif = YOLO(str(WEIGHTS_UNIF))
    lbl_root   = Path("/kaggle/input/datasets/mounir2mz/multiuav-yolo"
                      "/multiuav_yolo/labels/val")

    # Find densest sequence
    seq_dict = {}
    for p in sorted(MUAV_IMG_DIR.glob("*.jpg")):
        seq = p.stem.rsplit("_", 1)[0]
        seq_dict.setdefault(seq, []).append(p)
    for k in seq_dict: seq_dict[k] = sorted(seq_dict[k])

    best_seq, best_dens = None, 0
    for seq in list(seq_dict.keys())[:20]:
        total = sum(len([l for l in (lbl_root/(f.stem+".txt"))
                         .read_text().splitlines() if l.strip()])
                    for f in seq_dict[seq][:30]
                    if (lbl_root/(f.stem+".txt")).exists())
        dens = total / max(len(seq_dict[seq][:30]), 1)
        if dens > best_dens: best_dens, best_seq = dens, seq

    frames = seq_dict[best_seq]
    CAPTURE_IDX = min(25, len(frames)-1)

    bt = ByteTrack(); oc = OcSort()
    bt_res = oc_res = []
    capture = None

    for fi, fp in enumerate(frames[:CAPTURE_IDX+1]):
        img = cv2.imread(str(fp))
        if img is None: continue
        res = model_unif(img, imgsz=640, conf=0.25,
                         device=0, verbose=False)[0]
        det = np.empty((0,6), dtype=np.float32)
        if res.boxes and len(res.boxes) > 0:
            det = np.array([[*b.xyxy[0].cpu().numpy(),
                             float(b.conf[0]), 0]
                            for b in res.boxes], dtype=np.float32)
        bt_res = bt.update(det, img)
        oc_res = oc.update(det, img)
        if fi == CAPTURE_IDX: capture = img.copy()

    def get_color(tid):
        rng = np.random.default_rng(int(tid)*7+13)
        return tuple(int(c) for c in rng.integers(60,230,3))

    def draw(img, tracks, label):
        out = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for t in tracks:
            x1,y1,x2,y2 = map(int,t[:4]); tid=int(t[4])
            col=get_color(tid)
            cv2.rectangle(out,(x1,y1),(x2,y2),col,2)
            cv2.putText(out,str(tid),(x1,max(y1-4,10)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)
        return out

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Suivi multi-drones — {best_seq} "
                 f"(densité ≈{best_dens:.1f} UAVs/frame)\n"
                 f"Détecteur unifié YOLOv11s · modalité infrarouge",
                 fontsize=11, fontweight="bold")
    for ax, res, name in [
            (axes[0], bt_res, f"ByteTrack ({len(bt_res)} pistes)"),
            (axes[1], oc_res, f"OC-SORT ({len(oc_res)} pistes)")]:
        ax.imshow(draw(capture, res, name))
        ax.axis("off"); ax.set_title(name, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mot_examples.png",
                dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✅  mot_examples.png (sequence: {best_seq})")
except Exception as e:
    print(f"  ❌  {e}")

# ══════════════════════════════════════════════════════════════
# FIGURE 7 — Unified detector evaluation on RGBT (bar chart)
# ══════════════════════════════════════════════════════════════
print("\n[7/7] Unified detector cross-domain comparison ...")
try:
    domains  = ["DUT Anti-UAV\n(test)", "RGBT IR\n(val)", "RGBT Visible\n(val)"]
    map50    = [0.9177, 0.9929, 0.9853]
    map5095  = [0.6368, 0.6424, 0.6753]
    colors   = ["#2196F3", "#4CAF50", "#FF9800"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Détecteur unifié YOLOv11s — Performances cross-domaines",
                 fontsize=13, fontweight="bold")

    for ax, vals, metric in [
            (axes[0], map50, "mAP@0.50"),
            (axes[1], map5095, "mAP@0.50:0.95")]:
        bars = ax.bar(domains, vals, color=colors, edgecolor="white",
                      linewidth=0.5, width=0.5)
        ax.set_ylim(0.55, 1.02)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(metric, fontsize=11)
        ax.axhline(0.90, color="gray", linestyle="--",
                   alpha=0.6, label="Seuil 0.90")
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, val+0.005,
                    f"{val:.4f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "unified_detector_crossdomain.png",
                dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  ✅  unified_detector_crossdomain.png")
except Exception as e:
    print(f"  ❌  {e}")

# ── FINAL SUMMARY ─────────────────────────────────────────────
print("\n" + "="*60)
print("  FIGURE GENERATION COMPLETE")
print("="*60)
EXPECTED = [
    "confusion_matrix_specialist.png",
    "pr_curve_specialist.png",
    "training_curves_specialist.png",
    "detection_examples.png",
    "jitter_alphabeta.png",
    "tracker_states.png",
    "mot_examples.png",
    "unified_detector_crossdomain.png",
]
for f in EXPECTED:
    p = FIGURES_DIR / f
    ok = "✅" if p.exists() else "❌ MISSING"
    sz = f"({p.stat().st_size//1024} KB)" if p.exists() else ""
    print(f"  {ok}  {f} {sz}")
print(f"\n  All figures → {FIGURES_DIR}/")
