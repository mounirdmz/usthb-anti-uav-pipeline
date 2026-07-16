"""
╔══════════════════════════════════════════════════════════════╗
║  NOTEBOOK A — DATASET AUDIT                                  ║
║  USTHB Anti-UAV Thesis                                       ║
║                                                              ║
║  Audits all three datasets :                                 ║
║    1. DUT Anti-UAV      (RGB, SOT)                           ║
║    2. Anti-UAV RGBT     (IR + RGB, SOT)                      ║
║    3. MultiUAV_Train    (IR, MOT)                            ║
║                                                              ║
║  Outputs → /kaggle/working/audit/                            ║
║                                                              ║
║  KAGGLE DATASETS TO MOUNT :                                  ║
║    - mounir2mz/dut-anti-uav                                  ║
║    - mounir2mz/antiuav-rgbt-yolo                             ║
║    - mounir2mz/multiuav-yolo                                 ║
╚══════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import pandas as pd
import random
from pathlib import Path
from collections import defaultdict

OUT_DIR = Path("/kaggle/working/audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)
random.seed(42)
np.random.seed(42)

# ── BLOCK 1 : DUT Anti-UAV ────────────────────────────────────
print("=" * 60)
print("BLOCK 1 — DUT Anti-UAV")
print("=" * 60)

DUT_BASE = Path("/kaggle/input/datasets/mounir2mz/dut-anti-uav")
for split in ["train", "val", "test"]:
    imgs = list((DUT_BASE / split / split / "img").glob("*.jpg"))
    xmls = list((DUT_BASE / split / split / "xml").glob("*.xml"))
    print(f"  {split}: {len(imgs)} images, {len(xmls)} annotations")

widths, heights, areas, img_ratios, cx_list, cy_list = [], [], [], [], [], []
for xml_path in sorted((DUT_BASE / "train" / "train" / "xml").glob("*.xml")):
    try:
        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        if size is None: continue
        W, H = int(size.find("width").text), int(size.find("height").text)
        for obj in root.findall("object"):
            bb = obj.find("bndbox")
            if bb is None: continue
            xmin = float(bb.find("xmin").text)
            ymin = float(bb.find("ymin").text)
            xmax = float(bb.find("xmax").text)
            ymax = float(bb.find("ymax").text)
            w, h = xmax - xmin, ymax - ymin
            if w <= 0 or h <= 0: continue
            widths.append(w); heights.append(h)
            areas.append(w * h); img_ratios.append(w * h / (W * H))
            cx_list.append((xmin + w/2) / W)
            cy_list.append((ymin + h/2) / H)
    except Exception: continue

widths = np.array(widths); heights = np.array(heights)
areas  = np.array(areas);  img_ratios = np.array(img_ratios)

print(f"  Instances : {len(widths)}")
print(f"  Width  mean={widths.mean():.1f}  median={np.median(widths):.1f}")
print(f"  Height mean={heights.mean():.1f}  median={np.median(heights):.1f}")
print(f"  Area ratio mean={img_ratios.mean():.4f}")
print(f"  Small <32px : {100*np.mean((widths<32)|(heights<32)):.1f}%")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("DUT Anti-UAV — BBox Size Distribution", fontsize=13)
for ax, data, title, color in [
    (axes[0], widths,  "Width (px)",  "#4C72B0"),
    (axes[1], heights, "Height (px)", "#DD8452"),
    (axes[2], np.log10(areas+1), "log10(Area)", "#55A868")]:
    ax.hist(data, bins=60, color=color, edgecolor="none")
    if title != "log10(Area)":
        ax.axvline(32, color="orange", linestyle="--", label="32px"); ax.legend()
    ax.set_xlabel(title); ax.set_ylabel("Count"); ax.set_title(title)
plt.tight_layout()
plt.savefig(OUT_DIR / "dut_bbox_size.png", dpi=150, bbox_inches="tight")
plt.close(); print("  ✅ dut_bbox_size.png")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("DUT Anti-UAV — Spatial Center Distribution", fontsize=13)
hm, xe, ye = np.histogram2d(cx_list, cy_list, bins=50)
axes[0].imshow(hm.T, origin="lower", cmap="hot", aspect="auto")
axes[0].set_title("2D Density"); axes[0].set_xlabel("cx"); axes[0].set_ylabel("cy")
axes[1].hist(cx_list, bins=50, color="#4C72B0")
axes[1].axvline(0.5, color="red", linestyle="--", label="Center"); axes[1].legend()
axes[1].set_title("cx Distribution")
axes[2].hist(cy_list, bins=50, color="#55A868")
axes[2].axvline(0.5, color="red", linestyle="--", label="Center"); axes[2].legend()
axes[2].set_title("cy Distribution")
plt.tight_layout()
plt.savefig(OUT_DIR / "dut_center_distribution.png", dpi=150, bbox_inches="tight")
plt.close(); print("  ✅ dut_center_distribution.png")

thresholds = np.arange(8, 128, 2)
pct = [100*np.mean((widths<t)|(heights<t)) for t in thresholds]
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(thresholds, pct, color="#C44E52", linewidth=2)
for t in [16, 32, 64]:
    idx = np.argmin(np.abs(thresholds - t))
    ax.axvline(t, color="gray", linestyle="--", alpha=0.7)
    ax.annotate(f"{pct[idx]:.1f}%", xy=(t, pct[idx]), xytext=(t+2, pct[idx]+1), fontsize=9)
ax.set_xlabel("Size threshold (px)"); ax.set_ylabel("% BBoxes below threshold")
ax.set_title("DUT Anti-UAV — Small Object Prevalence"); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "dut_small_object_prevalence.png", dpi=150, bbox_inches="tight")
plt.close(); print("  ✅ dut_small_object_prevalence.png")

# ── BLOCK 2 : Anti-UAV RGBT ───────────────────────────────────
print("\n" + "=" * 60)
print("BLOCK 2 — Anti-UAV RGBT")
print("=" * 60)

RGBT_BASE = Path("/kaggle/input/datasets/mounir2mz/antiuav-rgbt-yolo/antiuav_rgbt_yolo")
SIZES = {"infrared": (640, 512), "visible": (1920, 1080)}
rgbt_stats = {}

for modality in ["infrared", "visible"]:
    img_w, img_h = SIZES[modality]
    lbl_dir = RGBT_BASE / modality / "labels" / "train"
    all_lbl = sorted(lbl_dir.glob("*.txt"))
    wl, hl, rl, opf = [], [], [], []
    empty = 0
    for lp in all_lbl:
        lines = [l.strip() for l in lp.read_text().splitlines() if l.strip()]
        opf.append(len(lines))
        if not lines: empty += 1; continue
        for line in lines:
            p = line.split()
            if len(p) != 5: continue
            _, cx, cy, nw, nh = map(float, p)
            wl.append(nw * img_w); hl.append(nh * img_h); rl.append(nw * nh)
    wa, ha, ra = np.array(wl), np.array(hl), np.array(rl)
    sp = 100*np.mean((wa<32)|(ha<32)) if len(wa)>0 else 0
    rgbt_stats[modality] = {"n": len(all_lbl), "inst": len(wa), "empty": empty,
                             "w": wa, "h": ha, "r": ra, "sp": sp,
                             "iw": img_w, "ih": img_h}
    print(f"\n  [{modality.upper()}] {img_w}×{img_h}")
    print(f"    Frames={len(all_lbl)}  Instances={len(wa)}  Empty={empty}({100*empty/max(len(all_lbl),1):.1f}%)")
    if len(wa)>0:
        print(f"    Width={wa.mean():.1f}px  Height={ha.mean():.1f}px  AreaRatio={ra.mean():.4f}  Small={sp:.1f}%")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Anti-UAV RGBT — BBox Distribution per Modality", fontsize=13)
for row, mod in enumerate(["infrared", "visible"]):
    s = rgbt_stats[mod]
    if len(s["w"]) == 0: continue
    color = "#4C72B0" if mod == "infrared" else "#DD8452"
    axes[row,0].hist(s["w"], bins=60, color=color)
    axes[row,0].axvline(32, color="orange", linestyle="--")
    axes[row,0].set_title(f"{mod.capitalize()} — Width")
    axes[row,1].hist(s["h"], bins=60, color=color)
    axes[row,1].axvline(32, color="orange", linestyle="--")
    axes[row,1].set_title(f"{mod.capitalize()} — Height")
plt.tight_layout()
plt.savefig(OUT_DIR / "rgbt_bbox_size.png", dpi=150, bbox_inches="tight")
plt.close(); print("\n  ✅ rgbt_bbox_size.png")

fig, ax = plt.subplots(figsize=(8, 4))
mods = ["infrared", "visible"]
vis = [100*(1-rgbt_stats[m]["empty"]/max(rgbt_stats[m]["n"],1)) for m in mods]
emp = [100-v for v in vis]
b1 = ax.bar(mods, vis, color=["#4C72B0","#DD8452"])
ax.bar(mods, emp, bottom=vis, color=["#aec6e8","#f5c6a0"])
for b, p in zip(b1, vis):
    ax.text(b.get_x()+b.get_width()/2, p/2, f"{p:.1f}%",
            ha="center", va="center", color="white", fontweight="bold")
ax.set_ylabel("% frames"); ax.set_title("Anti-UAV RGBT — Frame Visibility Rate")
plt.tight_layout()
plt.savefig(OUT_DIR / "rgbt_visibility_rate.png", dpi=150, bbox_inches="tight")
plt.close(); print("  ✅ rgbt_visibility_rate.png")

# ── BLOCK 3 : MultiUAV_Train ──────────────────────────────────
print("\n" + "=" * 60)
print("BLOCK 3 — MultiUAV_Train")
print("=" * 60)

MUAV_LBL = Path("/kaggle/input/datasets/mounir2mz/multiuav-yolo/multiuav_yolo/labels/train")
IMG_W, IMG_H = 640, 512
all_lbl = sorted(MUAV_LBL.glob("*.txt"))
wl, hl, al, rl, opf = [], [], [], [], []
empty_m = 0
seq_dict = defaultdict(list)

for lp in all_lbl:
    seq_dict[lp.stem.rsplit("_", 1)[0]].append(lp)
    lines = [l.strip() for l in lp.read_text().splitlines() if l.strip()]
    opf.append(len(lines))
    if not lines: empty_m += 1; continue
    for line in lines:
        p = line.split()
        if len(p) != 5: continue
        _, cx, cy, nw, nh = map(float, p)
        wp = nw*IMG_W; hp = nh*IMG_H
        wl.append(wp); hl.append(hp); al.append(wp*hp); rl.append(nw*nh)

wa = np.array(wl); ha = np.array(hl)
opf_a = np.array(opf); ra = np.array(rl)
sp_m = 100*np.mean((wa<32)|(ha<32))

print(f"  Sequences={len(seq_dict)}  Frames={len(all_lbl)}  Instances={len(wa)}")
print(f"  Empty={empty_m}  UAVs/frame mean={opf_a.mean():.1f}  max={opf_a.max()}")
print(f"  Width={wa.mean():.1f}px  Height={ha.mean():.1f}px")
print(f"  Area ratio={ra.mean():.4f}  Small<32px={sp_m:.1f}%")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("MultiUAV_Train — Statistics", fontsize=13)
axes[0].hist(wa, bins=60, color="#4C72B0"); axes[0].axvline(32, color="orange", linestyle="--")
axes[0].set_title("BBox Width"); axes[0].set_xlabel("px")
axes[1].hist(opf_a, bins=np.arange(0,opf_a.max()+2)-0.5, color="#C44E52")
axes[1].axvline(opf_a.mean(), color="black", linestyle="--", label=f"Mean={opf_a.mean():.1f}")
axes[1].set_title("UAVs per Frame"); axes[1].set_xlabel("Count"); axes[1].legend()
axes[2].hist(np.log10(np.array(al)+1), bins=60, color="#55A868")
axes[2].set_title("BBox Area (log scale)"); axes[2].set_xlabel("log10(px²)")
plt.tight_layout()
plt.savefig(OUT_DIR / "muav_statistics.png", dpi=150, bbox_inches="tight")
plt.close(); print("  ✅ muav_statistics.png")

# ── BLOCK 4 : Comparative Summary ─────────────────────────────
print("\n" + "=" * 60)
print("BLOCK 4 — Comparative Summary")
print("=" * 60)

summary = pd.DataFrame([
    {"Dataset": "DUT Anti-UAV", "Modality": "RGB", "Frames train": 10162,
     "Instances": len(widths), "UAVs/frame": 1.0,
     "Small <32px %": round(float(100*np.mean((widths<32)|(heights<32))),1),
     "Area ratio": round(float(img_ratios.mean()),4)},
    {"Dataset": "RGBT Infrared", "Modality": "IR",
     "Frames train": rgbt_stats["infrared"]["n"],
     "Instances": rgbt_stats["infrared"]["inst"], "UAVs/frame": 1.0,
     "Small <32px %": round(rgbt_stats["infrared"]["sp"],1),
     "Area ratio": round(float(rgbt_stats["infrared"]["r"].mean()),4)},
    {"Dataset": "RGBT Visible", "Modality": "RGB",
     "Frames train": rgbt_stats["visible"]["n"],
     "Instances": rgbt_stats["visible"]["inst"], "UAVs/frame": 1.0,
     "Small <32px %": round(rgbt_stats["visible"]["sp"],1),
     "Area ratio": round(float(rgbt_stats["visible"]["r"].mean()),4)},
    {"Dataset": "MultiUAV_Train", "Modality": "IR",
     "Frames train": len(all_lbl), "Instances": len(wa),
     "UAVs/frame": round(float(opf_a.mean()),1),
     "Small <32px %": round(float(sp_m),1),
     "Area ratio": round(float(ra.mean()),4)},
])
print(summary.to_string(index=False))
summary.to_csv(OUT_DIR / "dataset_summary.csv", index=False)
print("\n✅  Notebook A complete → /kaggle/working/audit/")
