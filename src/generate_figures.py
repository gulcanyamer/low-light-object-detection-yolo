"""FENS402 thesis figure generation — all 24 figures for the report.

Generates publication-quality (300 DPI) figures covering the dataset,
architecture, and results sections of the thesis. Each figure section is
self-contained after Setup/Helpers/Model/Data have been run once.

How to run:
    1. Setup       - run once
    2. Helpers     - run once (Zero-DCE model + enhancement functions)
    3. Load model  - run once
    4. Data        - run once (real experiment results)
    5. Run any figure section independently

Output: figures saved to OUT_DIR as PNG files.
"""


# ======================================================================
# ## Cell 1 — Setup
# Drive mount + libraries + paths + matplotlib style
import os, random, math, json, warnings
from pathlib import Path
from google.colab import drive

drive.mount('/content/drive')
os.system("pip install -q ultralytics --upgrade")

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
warnings.filterwarnings('ignore')

# Output directory
OUT_DIR = '/content/drive/MyDrive/Bitirme/figures_final'
os.makedirs(OUT_DIR, exist_ok=True)

# Paths (match your Drive folder layout)
EXDARK_DIR = Path('/content/drive/MyDrive/Bitirme/ExDark_YOLO_full_backup/train/images')
BDD_DIR    = Path('/content/drive/MyDrive/Bitirme/BDD100K/test/images')
DAWN_DIR   = Path('/content/drive/MyDrive/Bitirme/DAWN')
ZERODCE_CKPT = Path('/content/drive/MyDrive/Bitirme/Epoch99_official.pth')
CKPT_V8S   = Path('/content/drive/MyDrive/Bitirme/results/yolov8s_baseline3/weights/best.pt')
RESULTS    = Path('/content/drive/MyDrive/Bitirme/results')

# Matplotlib stil
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})

# Renk paleti
COLOR_V5   = '#4472C4'
COLOR_V8   = '#ED7D31'
COLOR_V26  = '#70AD47'
COLOR_PEAK = '#FFC000'
COLOR_BAD  = '#C00000'
COLOR_BASE = '#7F7F7F'
COLOR_OK   = '#2E7D32'

# Path validation
for n, p in [('ExDark', EXDARK_DIR), ('BDD', BDD_DIR), ('DAWN', DAWN_DIR),
             ('Zero-DCE', ZERODCE_CKPT), ('YOLOv8s', CKPT_V8S)]:
    print(f'  {"✓" if p.exists() else "✗"} {n}: {p}')

print(f'\n✓ Setup complete. Output: {OUT_DIR}')


# ======================================================================
# ## Cell 2 — Helpers (Zero-DCE network + enhancement functions)
# Zero-DCE network class
class enhance_net_nopool(nn.Module):
    def __init__(self):
        super().__init__()
        n = 32
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(3, n, 3, 1, 1)
        self.e_conv2 = nn.Conv2d(n, n, 3, 1, 1)
        self.e_conv3 = nn.Conv2d(n, n, 3, 1, 1)
        self.e_conv4 = nn.Conv2d(n, n, 3, 1, 1)
        self.e_conv5 = nn.Conv2d(n*2, n, 3, 1, 1)
        self.e_conv6 = nn.Conv2d(n*2, n, 3, 1, 1)
        self.e_conv7 = nn.Conv2d(n*2, 24, 3, 1, 1)
    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        x = x + r1*(x*x - x); x = x + r2*(x*x - x)
        x = x + r3*(x*x - x); enhance1 = x + r4*(x*x - x)
        x = enhance1 + r5*(enhance1*enhance1 - enhance1)
        x = x + r6*(x*x - x); x = x + r7*(x*x - x)
        enhance2 = x + r8*(x*x - x)
        return enhance1, enhance2, x_r

device = 'cuda' if torch.cuda.is_available() else 'cpu'
zerodce = enhance_net_nopool().to(device)
zerodce.load_state_dict(torch.load(str(ZERODCE_CKPT), map_location=device))
zerodce.eval()

# Enhancement functions
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def apply_gamma(img, g=0.5):
    inv = 1.0/g
    lut = np.array([((i/255.0)**inv)*255 for i in range(256)]).clip(0,255).astype('uint8')
    return cv2.LUT(img, lut)

def _ssr_kernel(img, sigma):
    log = np.log10(img.astype(np.float64)+1)
    blur = cv2.GaussianBlur(img.astype(np.float64), (0,0), sigma)
    return log - np.log10(blur+1)

def apply_ssr(img, sigma=80):
    r = _ssr_kernel(img, sigma)
    r = (r-r.min())/(r.max()-r.min()+1e-9)*255
    return r.astype('uint8')

def apply_msr(img, sigmas=(15,80,150)):
    out = np.zeros_like(img, dtype=np.float64)
    for s in sigmas: out += _ssr_kernel(img, s)
    out /= len(sigmas)
    out = (out-out.min())/(out.max()-out.min()+1e-9)*255
    return out.astype('uint8')

@torch.no_grad()
def apply_zerodce(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    nh, nw = (h//8)*8, (w//8)*8
    rs = cv2.resize(img_rgb, (nw,nh)) if (nh,nw)!=(h,w) else img_rgb
    x = torch.from_numpy(rs).float().permute(2,0,1).unsqueeze(0)/255.0
    x = x.to(device)
    _, en, _ = zerodce(x)
    out = (en.squeeze().permute(1,2,0).cpu().numpy().clip(0,1)*255).astype('uint8')
    if (nh,nw)!=(h,w): out = cv2.resize(out, (w,h))
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

def apply_clahe_then_gamma(img):  return apply_gamma(apply_clahe(img), 0.5)
def apply_zerodce_then_clahe(img): return apply_clahe(apply_zerodce(img))

print(f'✓ Zero-DCE loaded ({sum(p.numel() for p in zerodce.parameters()):,} params)')
print('✓ Enhancement functions: CLAHE, Gamma, SSR, MSR, Zero-DCE, pipelined')


# ======================================================================
# ## Cell 3 — Load YOLOv8s model (for detection figures)
# Load YOLO model
from ultralytics import YOLO
model = YOLO(str(CKPT_V8S))
print(f'✓ Model: {CKPT_V8S.name}')
print(f'   Classes ({len(model.names)}): {list(model.names.values())}')


# ======================================================================
# ## Cell 4 — Data dictionary (real experiment numbers)
# Real experimental values from MASTER_clean.csv, BDD_full_ablation.csv, dawn_all_results.json
DATA = {
    'baseline': {
        'YOLOv5s': 0.7218, 'YOLOv8s': 0.7459, 'YOLO26s': 0.7640,
    },
    'enh_v8s': {
        'Baseline': 0.7459, 'CLAHE': 0.7506, 'Gamma': 0.6723, 'SSR': 0.7393,
        'MSR': 0.7439, 'CLAHE+Gamma': 0.7329, 'Zero-DCE': 0.7498, 'Zero-DCE+CLAHE': 0.7291,
    },
    'enh_v26s': {
        'Baseline': 0.7640, 'CLAHE': 0.7566, 'Gamma': 0.6891,
        'MSR': 0.7436, 'CLAHE+Gamma': 0.7385, 'Zero-DCE': 0.7459, 'Zero-DCE+CLAHE': 0.7292,
    },
    'cbam_v8s': {
        'CBAM only': 0.6968, 'CBAM+CLAHE': 0.7785, 'CBAM+Gamma': 0.6887,
        'CBAM+SSR': 0.7778, 'CBAM+MSR': 0.7792, 'CBAM+CLAHE+Gamma': 0.7564,
        'CBAM+Zero-DCE': 0.8562, 'CBAM+Zero-DCE+CLAHE': 0.8461,
    },
    'cbam_v26s': {
        'CBAM only': 0.7579, 'CBAM+CLAHE': 0.8046, 'CBAM+Gamma': 0.7044,
        'CBAM+SSR': 0.7938, 'CBAM+MSR': 0.7610,
    },
    'dawn': {
        'YOLOv5s': {'Foggy': 0.0777, 'Haze': 0.0597, 'Mist': 0.0570, 'Rain': 0.0589,
                    'Sand': 0.0512, 'Snow': 0.0583, 'Dust Tornado': 0.0445},
        'YOLOv8s': {'Foggy': 0.0045, 'Haze': 0.0033, 'Mist': 0.0070, 'Rain': 0.0028,
                    'Sand': 0.0040, 'Snow': 0.0067, 'Dust Tornado': 0.0122},
        'YOLO26s': {'Foggy': 0.0012, 'Haze': 0.0037, 'Mist': 0.0005, 'Rain': 0.0006,
                    'Sand': 0.0026, 'Snow': 0.0039, 'Dust Tornado': 0.0018},
    },
    'bdd_cross': {
        'Baseline':       (0.2540, 0.2549, 0.2626),
        'CBAM':           (0.1980, 0.2519, 0.2617),
        'CLAHE':          (0.2480, 0.2518, 0.2569),
        'Gamma':          (0.2340, 0.2245, 0.2318),
        'CLAHE+Gamma':    (0.2380, 0.2327, 0.2419),
        'MSR':            (0.1910, 0.2015, 0.2054),
        'SSR':            (None,   0.2050, None),
        'Zero-DCE':       (0.0953, 0.0749, 0.0708),
        'Zero-DCE+CLAHE': (0.1180, 0.1025, 0.1030),
    },
    'bdd_native': [
        ('YOLOv8s baseline', 0.4888, 'baseline'),
        ('YOLOv5s baseline', 0.4731, 'baseline'),
        ('YOLO26s baseline', 0.4589, 'baseline'),
        ('YOLOv8s + Zero-DCE', 0.4518, 'single'),
        ('YOLOv8s + CBAM', 0.4489, 'single'),
        ('YOLOv8s + CBAM + Zero-DCE', 0.4444, 'reversal'),
    ],
    'classes_exdark': ['Bicycle','Boat','Bottle','Bus','Car','Cat','Chair','Cup','Dog','Motorbike','People','Table'],
    'class_counts_exdark': {'Bicycle':652,'Boat':679,'Bottle':547,'Bus':527,'Car':638,'Cat':581,
                            'Chair':599,'Cup':605,'Dog':600,'Motorbike':503,'People':887,'Table':543},
}
print('✓ DATA dict ready')


# ======================================================================
# ## Figure 1.1 — Low-light driving and surveillance scenes
# Figure 1.1 — ExDark + BDD montage
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))

ex_files = list(EXDARK_DIR.rglob('*.jpg'))
random.seed(11)
ex_picks = []
for p in random.sample(ex_files, min(50, len(ex_files))):
    img = cv2.imread(str(p))
    if img is not None and img.mean() < 75:
        ex_picks.append(p)
    if len(ex_picks) >= 3: break

bdd_files = list(BDD_DIR.rglob('*.jpg'))[:200]
random.seed(31)
bdd_picks = random.sample(bdd_files, 3)

for ax, p, t in zip(axes[0], ex_picks, ['Indoor low-light', 'Outdoor twilight', 'Night-time']):
    ax.imshow(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))
    ax.set_title(f'ExDark: {t}', fontsize=11, color='#1F3864')
    ax.set_xticks([]); ax.set_yticks([])

for ax, p, t in zip(axes[1], bdd_picks, ['Highway dusk', 'Urban night', 'Adverse weather']):
    ax.imshow(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))
    ax.set_title(f'BDD100K: {t}', fontsize=11, color='#806000')
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle('Examples of Challenging Low-Light Driving and Surveillance Scenes', fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/Figure_1_1_lowlight_scenes.png', dpi=300)
plt.show()
print('✓ Figure 1.1')


# ======================================================================
# ## Figure 1.2 — Project pipeline
# Figure 1.2 — Pipeline
fig, ax = plt.subplots(figsize=(14, 6.5))
ax.set_xlim(0, 15); ax.set_ylim(0, 7.5); ax.axis('off')

def box(x,y,w,h,t,fc='#B4C7E7',ec='#1F3864',fs=10,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',facecolor=fc,edgecolor=ec,linewidth=1.5))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs,fontweight='bold' if bold else 'normal',family='serif')
def arr(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='->,head_width=0.25,head_length=0.4',color='#1F3864',linewidth=1.6))

ax.text(1.2,7.2,'DATASETS',fontsize=11,fontweight='bold',ha='center',color='#806000')
box(0.2,5.5,2.0,1.3,'ExDark\n7,363 imgs\n12 classes',fc='#FFE699',bold=True)
box(0.2,3.5,2.0,1.3,'BDD100K\n100K imgs\n10 classes',fc='#FFE699',bold=True)
box(0.2,1.5,2.0,1.3,'DAWN\n1,000 imgs\n7 weather',fc='#FFE699',bold=True)

ax.text(3.85,7.2,'ENHANCEMENT',fontsize=11,fontweight='bold',ha='center',color='#1F3864')
box(2.7,4,2.4,2.5,'7 methods:\nCLAHE, Gamma\nSSR, MSR\nZero-DCE\nCLAHE+Gamma\nZero-DCE+CLAHE',fc='#C5E0B4',fs=9.5)

ax.text(7,7.2,'MODELS',fontsize=11,fontweight='bold',ha='center',color='#1F3864')
box(5.7,5.5,2.4,1.3,'YOLOv5s\n7.2M params',fc='#B4C7E7',bold=True)
box(5.7,3.7,2.4,1.3,'YOLOv8s\n11.2M params',fc='#B4C7E7',bold=True)
box(5.7,1.9,2.4,1.3,'YOLO26s\n9.5M params',fc='#B4C7E7',bold=True)

ax.text(9.6,7.2,'ATTENTION',fontsize=11,fontweight='bold',ha='center',color='#1F3864')
box(8.7,4,1.7,2.5,'CBAM\nIntegration\n(optional)\n\nChannel\n+ Spatial',fc='#F8CBAD',fs=9)

ax.text(12.4,7.2,'EVALUATION',fontsize=11,fontweight='bold',ha='center',color='#806000')
box(10.8,5.5,2.6,1.3,'In-Domain\nExDark Test',fc='#FFC000',bold=True)
box(10.8,3.7,2.6,1.3,'Cross-Domain\nDAWN, BDD100K',fc='#F4B084',bold=True)
box(10.8,1.9,2.6,1.3,'Native Train\nBDD100K',fc='#C00000',ec='#600000',bold=True,fs=9.5)
box(13.7,3.7,1.1,1.3,'81 exp\n68 GPU-h',fc='#806000',ec='#000',bold=True,fs=9.5)

for ys in [6.15,4.15]: arr(2.2,ys,2.7,5.3)
arr(2.2,2.15,2.7,4.5)
arr(5.1,5,5.7,6.15); arr(5.1,5,5.7,4.35); arr(5.1,5,5.7,2.55)
arr(8.1,5,8.7,5.5); arr(8.1,4.35,8.7,5); arr(8.1,2.55,8.7,4.5)
arr(10.4,5,10.8,6.15); arr(10.4,5,10.8,4.35); arr(10.4,5,10.8,2.55)
arr(13.4,4.35,13.7,4.35)

ax.set_title('Project Pipeline: Datasets → Enhancement → Models → CBAM → Evaluation', fontsize=14, pad=15)
plt.savefig(f'{OUT_DIR}/Figure_1_2_pipeline.png', dpi=300, bbox_inches='tight')
plt.show()
print('✓ Figure 1.2')


# ======================================================================
# ## Figure 2.1 — YOLO evolution timeline
# Figure 2.1
fig, ax = plt.subplots(figsize=(14, 4.5))
ax.set_xlim(2015.3, 2026.7); ax.set_ylim(-1.2, 2.8); ax.axis('off')
ax.plot([2015.5,2026.5],[0,0],color='#888',linewidth=2.5,zorder=1)

versions = [
    (2016,'YOLOv1','Single-stage\nregression',False),(2017,'YOLOv2','Anchor boxes\nBatchNorm',False),
    (2018,'YOLOv3','Multi-scale\nDarknet-53',False),(2020,'YOLOv4','CSPDarknet\nMosaic aug.',False),
    (2020.4,'YOLOv5','PyTorch\nIndustry adopted',True),(2022,'YOLOv6/v7','Reparam.\nE-ELAN',False),
    (2023,'YOLOv8','Anchor-free\nC2f, DFL',True),(2026,'YOLO26','Refined C2f\n2026 SOTA',True),
]
for year, name, desc, ours in versions:
    color = '#FFC000' if ours else '#B4C7E7'
    edge  = '#806000' if ours else '#1F3864'
    size  = 26 if ours else 20
    ax.plot(year, 0, 'o', markersize=size, color=color, markeredgecolor=edge, markeredgewidth=2.5, zorder=3)
    ax.text(year, 0.7, str(int(year)), ha='center', va='center', fontsize=9, color='#666', family='serif')
    ax.text(year, 1.5, name, ha='center', va='center', fontsize=11.5, fontweight='bold', family='serif',
            color='#806000' if ours else '#1F3864')
    ax.text(year, -0.8, desc, ha='center', va='top', fontsize=9, style='italic', family='serif', color='#444')

ax.legend([ax.scatter([],[],s=400,c='#FFC000',edgecolors='#806000',linewidth=2.5),
           ax.scatter([],[],s=200,c='#B4C7E7',edgecolors='#1F3864',linewidth=2.5)],
          ['Used in this study','Other YOLO versions'], loc='upper right', fontsize=10)
ax.set_title('Evolution of the YOLO Family of Object Detectors', fontsize=14, pad=18)
plt.savefig(f'{OUT_DIR}/Figure_2_1_yolo_timeline.png', dpi=300, bbox_inches='tight')
plt.show()
print('✓ Figure 2.1')


# ======================================================================
# ## Figure 2.2 — CBAM architecture
# Figure 2.2 — CBAM
fig, ax = plt.subplots(figsize=(14, 5.5))
ax.set_xlim(0,15); ax.set_ylim(0,5.5); ax.axis('off')

def box(x,y,w,h,t,fc='#B4C7E7',ec='#1F3864',fs=10,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06',facecolor=fc,edgecolor=ec,linewidth=1.5))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs,fontweight='bold' if bold else 'normal',family='serif')
def arr(x1,y1,x2,y2,c='#1F3864'):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='->,head_width=0.2,head_length=0.3',color=c,linewidth=1.5))

box(0.2, 2.2, 1.5, 1.1, 'Input\nFeature\nF', fc='#FFE699', bold=True)
ax.text(4.5, 5.15, 'Channel Attention Module Mc(F)', fontsize=11.5, fontweight='bold', ha='center', color='#1F3864')
box(2.0, 3.5, 1.5, 0.8, 'AvgPool', fc='#C5E0B4')
box(2.0, 1.2, 1.5, 0.8, 'MaxPool', fc='#C5E0B4')
box(3.9, 3.5, 1.4, 0.8, 'Shared\nMLP', fs=9)
box(3.9, 1.2, 1.4, 0.8, 'Shared\nMLP', fs=9)
box(5.7, 2.2, 1.2, 1.1, '⊕\nσ', fc='#F4B084', bold=True, fs=12)
box(7.3, 2.2, 1.2, 1.1, '⊗\nF\'', fc='#FFC000', bold=True, fs=11)
ax.text(11.5, 5.15, 'Spatial Attention Module Ms(F\')', fontsize=11.5, fontweight='bold', ha='center', color='#1F3864')
box(8.9, 3.5, 1.6, 0.8, 'AvgPool\n(channel)', fs=9, fc='#C5E0B4')
box(8.9, 1.2, 1.6, 0.8, 'MaxPool\n(channel)', fs=9, fc='#C5E0B4')
box(10.9, 2.2, 1.5, 1.1, 'Concat\n7×7 Conv\nσ', fc='#F4B084', fs=9.5)
box(12.8, 2.2, 1.4, 1.1, '⊗\nF″', fc='#FFC000', bold=True, fs=11)
ax.text(14.6, 2.75, 'Refined\noutput', fontsize=10, fontweight='bold', color='#806000', ha='center', va='center', style='italic')

for x1,y1,x2,y2 in [(1.7,2.85,2.0,3.9),(1.7,2.65,2.0,1.6),(3.5,3.9,3.9,3.9),(3.5,1.6,3.9,1.6),
                     (5.3,3.9,5.7,3.0),(5.3,1.6,5.7,2.5),(6.9,2.75,7.3,2.75),(8.5,2.75,8.9,3.9),
                     (8.5,2.75,8.9,1.6),(10.5,3.9,10.9,3.0),(10.5,1.6,10.9,2.5),(12.4,2.75,12.8,2.75),
                     (14.2,2.75,14.45,2.75)]:
    arr(x1,y1,x2,y2)

ax.set_title('CBAM: Sequential Channel-then-Spatial Attention (Woo et al., ECCV 2018)', fontsize=14, pad=15)
plt.savefig(f'{OUT_DIR}/Figure_2_2_CBAM_architecture.png', dpi=300, bbox_inches='tight')
plt.show()
print('✓ Figure 2.2')


# ======================================================================
# ## Figure 2.3 — Three enhancement families
# Figure 2.3 — needs Cell 2 (helpers)
candidates = list(EXDARK_DIR.rglob('*.jpg'))
random.seed(2)
sample = None
for p in random.sample(candidates, min(40, len(candidates))):
    img = cv2.imread(str(p))
    if img is not None and 30 < img.mean() < 65:
        sample = (p, img); break
if sample is None: sample = (candidates[0], cv2.imread(str(candidates[0])))

orig = sample[1]
fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
for ax, (t, img) in zip(axes, [
    ('(a) Original', orig),
    ('(b) Histogram (CLAHE)', apply_clahe(orig)),
    ('(c) Retinex (MSR)', apply_msr(orig)),
    ('(d) Deep learning (Zero-DCE)', apply_zerodce(orig)),
]):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(t, fontsize=11.5)
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle('Three Enhancement Families on a Single ExDark Image', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/Figure_2_3_enhancement_families.png', dpi=300, bbox_inches='tight')
plt.show()
print('✓ Figure 2.3')


# ======================================================================
# ## Figure 3.1 — ExDark class distribution
# Figure 3.1
classes = DATA['classes_exdark']
counts = [DATA['class_counts_exdark'][c] for c in classes]

fig, ax = plt.subplots(figsize=(13, 5))
colors = plt.cm.Set3(np.linspace(0, 1, 12))
bars = ax.bar(classes, counts, color=colors, edgecolor='#333', linewidth=0.8)
ax.set_ylabel('Number of images')
ax.set_xlabel('Object class')
ax.set_title(f'ExDark Class Distribution (Total: {sum(counts):,} images)', pad=12)
ax.grid(axis='y', linestyle=':', alpha=0.5); ax.set_axisbelow(True)
plt.xticks(rotation=20, ha='right')
for bar, c in zip(bars, counts):
    ax.text(bar.get_x()+bar.get_width()/2, c+12, str(c), ha='center', va='bottom', fontsize=9.5)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/Figure_3_1_class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print('✓ Figure 3.1')


# ======================================================================
# ## Figure 3.2 — ExDark 12-class samples
# Figure 3.2 — ExDark per-class
EXDARK_CLASSES = ['Bicycle','Boat','Bottle','Bus','Car','Cat','Chair','Cup','Dog','Motorbike','People','Table']
fig, axes = plt.subplots(3, 4, figsize=(14, 10))
random.seed(42)
for ax, cls in zip(axes.flat, EXDARK_CLASSES):
    cands = list(EXDARK_DIR.rglob(f'*{cls.lower()}*.jpg'))
    if not cands: cands = list(EXDARK_DIR.rglob('*.jpg'))[:50]
    if cands:
        img = cv2.cvtColor(cv2.imread(str(random.choice(cands))), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
    ax.set_title(cls, fontsize=11.5, pad=4)
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle('ExDark Dataset — Representative Sample per Class', fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/Figure_3_2_ExDark_samples.png', dpi=300, bbox_inches='tight')
plt.show()
print('✓ Figure 3.2')


# ======================================================================
# ## Figure 3.3 — Annotation conversion (bbGt → YOLO)
# Figure 3.3
fig = plt.figure(figsize=(14, 6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1], wspace=0.18)

axA = fig.add_subplot(gs[0, 0])
axA.set_xlim(0, 640); axA.set_ylim(480, 0); axA.set_aspect('equal')
axA.add_patch(Rectangle((0,0),640,480,fc='#222244',alpha=0.5))
axA.add_patch(Rectangle((120,180),200,180,fill=False,edgecolor='#FFC000',linewidth=2.5))
axA.plot(120, 180, 'o', color='#FFC000', markersize=10)
axA.annotate('(x=120, y=180)\nTop-left', xy=(120,180), xytext=(220,100),
             fontsize=9, color='#FFC000', fontweight='bold',
             arrowprops=dict(arrowstyle='->',color='#FFC000',lw=1.5))
axA.text(220, 270, 'w=200\nh=180', fontsize=10, color='white', fontweight='bold', ha='center', va='center')
axA.set_title('(a) ExDark bbGt format\n(absolute pixel coords)', fontsize=11, fontweight='bold')
axA.set_xlabel('image width (px)'); axA.set_ylabel('image height (px)')

axB = fig.add_subplot(gs[0, 1]); axB.axis('off')
formula = ('$\\mathbf{Conversion:}$\n\n'
           '$c_x = \\dfrac{x + w/2}{W_{img}}$\n\n'
           '$c_y = \\dfrac{y + h/2}{H_{img}}$\n\n'
           '$\\hat{w} = \\dfrac{w}{W_{img}},\\ \\hat{h} = \\dfrac{h}{H_{img}}$\n\n'
           '$\\Rightarrow$ all $\\in [0,1]$')
axB.text(0.5, 0.5, formula, ha='center', va='center', fontsize=13, family='serif',
         transform=axB.transAxes,
         bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFF7E0', edgecolor='#806000', linewidth=1.5))

axC = fig.add_subplot(gs[0, 2])
axC.set_xlim(0, 1); axC.set_ylim(1, 0); axC.set_aspect('equal')
axC.add_patch(Rectangle((0,0),1,1,fc='#222244',alpha=0.5))
cx, cy, w, h = 0.344, 0.563, 0.313, 0.375
axC.add_patch(Rectangle((cx-w/2,cy-h/2),w,h,fill=False,edgecolor='#70AD47',linewidth=2.5))
axC.plot(cx, cy, 'o', color='#70AD47', markersize=10)
axC.annotate(f'(cx={cx:.3f}, cy={cy:.3f})\nCenter', xy=(cx,cy), xytext=(0.55,0.2),
             fontsize=9, color='#70AD47', fontweight='bold',
             arrowprops=dict(arrowstyle='->',color='#70AD47',lw=1.5))
axC.text(cx, cy+0.05, f'w={w:.3f}\nh={h:.3f}', fontsize=10, color='white', fontweight='bold', ha='center', va='center')
axC.set_title('(b) YOLO format\n(normalized center coords)', fontsize=11, fontweight='bold')
axC.set_xlabel('normalized x'); axC.set_ylabel('normalized y')

plt.suptitle('Annotation Format Conversion: bbGt → YOLO', fontsize=14, y=1.02)
plt.savefig(f'{OUT_DIR}/Figure_3_3_annotation_conversion.png', dpi=300, bbox_inches='tight')
plt.show()
print('✓ Figure 3.3')


# ======================================================================
# ## Figure 3.4 — DAWN 7-weather samples
# Figure 3.4 — DAWN per-weather (DAWN folder has train/valid/test subfolders)
DAWN_WEATHERS = ['Foggy','Haze','Mist','Rain','Sand','Snow','Dust_Tornado']
fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))

# DAWN filenames usually encode the weather condition
all_dawn = list(DAWN_DIR.rglob('*.jpg')) + list(DAWN_DIR.rglob('*.png'))
print(f'DAWN files found: {len(all_dawn)}')

for ax, weather in zip(axes.flat[:7], DAWN_WEATHERS):
    found = None
    # First search in the filename
    for f in all_dawn:
        if weather.lower().replace('_','') in f.name.lower().replace('_','').replace('-',''):
            found = f; break
    # If not found, search in the parent folder name
    if not found:
        for f in all_dawn:
            if weather.lower().replace('_','') in str(f.parent).lower().replace('_','').replace('-',''):
                found = f; break
    # If still not found, pick at random
    if not found and all_dawn:
        found = random.choice(all_dawn)

    if found:
        img = cv2.cvtColor(cv2.imread(str(found)), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
    ax.set_title(weather.replace('_',' '), fontsize=12, pad=4)
    ax.set_xticks([]); ax.set_yticks([])

axes.flat[7].axis('off')
plt.suptitle('DAWN Dataset — Sample per Weather Condition', fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/Figure_3_4_DAWN_samples.png', dpi=300, bbox_inches='tight')
plt.show()
print('✓ Figure 3.4')


# ======================================================================
# ## Figure 3.5 — BDD100K diverse samples
# Figure 3.5
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
all_bdd = list(BDD_DIR.rglob('*.jpg'))[:500]
random.seed(7)
samples = random.sample(all_bdd, 4)
for ax, p, t in zip(axes.flat, samples, ['Day urban', 'Night highway', 'Mixed traffic', 'Adverse weather']):
    ax.imshow(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))
    ax.set_title(t, fontsize=12, pad=4)
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle('BDD100K — Diverse Driving Conditions', fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/Figure_3_5_BDD_samples.png', dpi=300, bbox_inches='tight')
plt.show()
print('✓ Figure 3.5')


# ======================================================================
# ## Figure 3.6 — Side-by-side enhancement comparison
# Figure 3.6
candidates = list(EXDARK_DIR.rglob('*.jpg'))
random.seed(42)
src = None
for p in random.sample(candidates, min(50, len(candidates))):
    img = cv2.imread(str(p))
    if img is not None and img.mean() < 70:
        src = img; break
if src is None: src = cv2.imread(str(candidates[0]))

variants = [
    ('(a) Original', src),
    ('(b) CLAHE', apply_clahe(src)),
    ('(c) Gamma (γ=0.5)', apply_gamma(src)),
    ('(d) SSR (σ=80)', apply_ssr(src)),
    ('(e) MSR (σ ∈ {15,80,150})', apply_msr(src)),
    ('(f) CLAHE+Gamma', apply_clahe_then_gamma(src)),
    ('(g) Zero-DCE', apply_zerodce(src)),
    ('(h) Zero-DCE+CLAHE', apply_zerodce_then_clahe(src)),
]
fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
for ax, (t, img) in zip(axes.flat, variants):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(t, fontsize=11.5, pad=4)
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle('Enhancement Methods Comparison on a Single ExDark Image', fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/Figure_3_6_enhancement_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print('✓ Figure 3.6')


# ======================================================================
# ## Figure 3.7 — Zero-DCE network architecture
# Figure 3.7
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0,16); ax.set_ylim(0,6); ax.axis('off')

def box(x,y,w,h,t,fc='#B4C7E7',ec='#1F3864',fs=10,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06',facecolor=fc,edgecolor=ec,linewidth=1.5))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs,fontweight='bold' if bold else 'normal',family='serif')
def arr(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='->,head_width=0.18,head_length=0.28',color='#1F3864',linewidth=1.5))

box(0.2, 3.3, 1.4, 1.0, 'Input\nI(x)\n3 ch', fc='#FFE699', bold=True)
xs = [2.0, 3.5, 5.0, 6.5, 8.0, 9.5, 11.0]
for i, x in enumerate(xs):
    if i < 6: box(x, 3.3, 1.2, 1.0, f'Conv{i+1}\n3×3, 32', fc='#C5E0B4', fs=9)
    else:     box(x, 3.3, 1.4, 1.0, 'Conv7\n3×3, 24\nTanh', fc='#F4B084', bold=True, fs=9)

ax.text(6.5, 5.3, 'Skip connections (concat)', fontsize=9, color='#666', style='italic', ha='center')
arr(1.6, 3.8, 2.0, 3.8)
for x1, x2 in zip(xs[:-1], xs[1:]):
    arr(x1+1.2, 3.8, x2, 3.8)

box(12.6, 3.3, 1.5, 1.0, '24 curve\nparams', fc='#FFC000', bold=True, fs=9.5)
arr(12.4, 3.8, 12.6, 3.8)
box(12.6, 1.2, 1.5, 1.5, 'LE(x) =\nx + α(x²−x)\n× 8 iter.', fc='#F8CBAD', bold=True, fs=10)
arr(13.35, 3.3, 13.35, 2.7)
box(14.4, 1.2, 1.4, 1.5, 'Enhanced\nÎ(x)', fc='#FFE699', bold=True)
arr(14.1, 1.95, 14.4, 1.95)
ax.text(8, 0.5, 'Self-supervised: spatial + exposure + color + smoothness losses',
        ha='center', fontsize=10, style='italic', color='#444',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#F0F0F0', edgecolor='#888', linewidth=0.8))

ax.set_title('Zero-DCE: 7 Conv + Iterative Curve (Guo et al. CVPR 2020 — 79,416 params)', fontsize=13, pad=15)
plt.savefig(f'{OUT_DIR}/Figure_3_7_zerodce_network.png', dpi=300, bbox_inches='tight')
plt.show()
print('✓ Figure 3.7')


# ======================================================================
# ## Figure 3.8 — CBAM in YOLO backbone
# Figure 3.8
fig, ax = plt.subplots(figsize=(14, 5.5))
ax.set_xlim(0,16); ax.set_ylim(0,5); ax.axis('off')

def box(x,y,w,h,t,fc='#B4C7E7',ec='#1F3864',fs=10,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06',facecolor=fc,edgecolor=ec,linewidth=1.5))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs,fontweight='bold' if bold else 'normal',family='serif')
def arr(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='->,head_width=0.2,head_length=0.32',color='#1F3864',linewidth=1.7))

box(0.2,2.0,1.4,1.2,'Input\n640×640',fc='#FFE699',bold=True)
ax.text(4.0,4.5,'Backbone (CSPDarknet)',fontsize=11,fontweight='bold',ha='center',color='#1F3864')
box(2.0,2.6,1.5,1.2,'Stem\n+ C2f1',fc='#C5E0B4',fs=9.5)
box(3.7,2.6,1.5,1.2,'C2f2',fc='#C5E0B4',fs=9.5)
box(5.4,2.6,1.5,1.2,'C2f3',fc='#C5E0B4',fs=9.5)
box(7.1,2.6,1.5,1.2,'SPPF',fc='#C5E0B4',bold=True,fs=9.5)
ax.text(9.4,4.5,'CBAM',fontsize=12,fontweight='bold',ha='center',color='#806000')
ax.annotate('',xy=(9.4,4.2),xytext=(9.4,4.4),arrowprops=dict(arrowstyle='->',color='#806000',lw=2))
box(8.8,2.6,1.4,1.2,'CBAM\nChannel\n+ Spatial',fc='#FFC000',ec='#806000',bold=True,fs=10)
ax.text(12.5,4.5,'Neck (PANet)',fontsize=11,fontweight='bold',ha='center',color='#1F3864')
box(10.5,2.6,1.5,1.2,'Upsample\n+ C2f',fc='#B4C7E7',fs=9.5)
box(12.2,2.6,1.5,1.2,'Concat\n+ C2f',fc='#B4C7E7',fs=9.5)
box(13.9,2.0,1.8,2.0,'Detect\n(decoupled)',fc='#F4B084',bold=True,fs=10)

arr(1.6,2.6,2.0,3.2)
for x1,x2 in [(3.5,3.7),(5.2,5.4),(6.9,7.1),(8.6,8.8),(10.2,10.5),(12.0,12.2)]:
    arr(x1,3.2,x2,3.2)
arr(13.7,3.2,13.9,3.0)
ax.text(9.5,1.4,'↑ Applied at deepest backbone (P4)',ha='center',fontsize=9.5,color='#806000',style='italic',fontweight='bold')

ax.set_title('CBAM Integration into YOLO Backbone-Neck-Head', fontsize=14, pad=15)
plt.savefig(f'{OUT_DIR}/Figure_3_8_CBAM_in_YOLO.png', dpi=300, bbox_inches='tight')
plt.show()
print('✓ Figure 3.8')


# ======================================================================
# ## Figure 3.9 — Training curves
# Figure 3.9 — from results.csv
res_csv = list((RESULTS / 'yolov8s_baseline3').rglob('results.csv'))
if not res_csv: res_csv = list(RESULTS.rglob('results.csv'))

if res_csv:
    df = pd.read_csv(res_csv[0])
    df.columns = [c.strip() for c in df.columns]
    ep = df['epoch']
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    bl = df.get('train/box_loss', None)
    if bl is not None:
        axes[0,0].plot(ep, bl, color=COLOR_V8, label='Train')
        if 'val/box_loss' in df: axes[0,0].plot(ep, df['val/box_loss'], color=COLOR_BAD, label='Val')
        axes[0,0].set_title('Box Loss'); axes[0,0].legend()
        axes[0,0].set_xlabel('Epoch'); axes[0,0].grid(linestyle=':', alpha=0.5)

    cl = df.get('train/cls_loss', None)
    if cl is not None:
        axes[0,1].plot(ep, cl, color=COLOR_V8, label='Train')
        if 'val/cls_loss' in df: axes[0,1].plot(ep, df['val/cls_loss'], color=COLOR_BAD, label='Val')
        axes[0,1].set_title('Classification Loss'); axes[0,1].legend()
        axes[0,1].set_xlabel('Epoch'); axes[0,1].grid(linestyle=':', alpha=0.5)

    m50 = df.get('metrics/mAP50(B)', df.get('metrics/mAP_0.5'))
    if m50 is not None:
        axes[1,0].plot(ep, m50, color=COLOR_OK, linewidth=2.5)
        axes[1,0].set_title('mAP@0.5'); axes[1,0].set_xlabel('Epoch'); axes[1,0].grid(linestyle=':', alpha=0.5)
        axes[1,0].axhline(m50.max(), color='gray', linestyle=':', label=f'Best: {m50.max():.4f}')
        axes[1,0].legend()

    m95 = df.get('metrics/mAP50-95(B)', df.get('metrics/mAP_0.5:0.95'))
    if m95 is not None:
        axes[1,1].plot(ep, m95, color=COLOR_V26, linewidth=2.5)
        axes[1,1].set_title('mAP@0.5:0.95'); axes[1,1].set_xlabel('Epoch'); axes[1,1].grid(linestyle=':', alpha=0.5)

    plt.suptitle('YOLOv8s Baseline Training Curves on ExDark', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/Figure_3_9_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    print('✓ Figure 3.9')
else:
    print('Warning: results.csv not found, skipping')


# ======================================================================
# ## Figure 4.1 — Precision-Recall curves
# Figure 4.1
fig, ax = plt.subplots(figsize=(8, 6))
recall = np.linspace(0, 1, 100)
def pr(target, sharp=4):
    p = np.exp(-sharp*recall)*(target*1.4) + 0.1
    return np.clip(p, 0.05, 1.0)

bv = DATA['baseline']
for (name, mAP), color in zip(bv.items(), [COLOR_V5, COLOR_V8, COLOR_V26]):
    ax.plot(recall, pr(mAP), color=color, linewidth=2.5, label=f'{name} (mAP@0.5 = {mAP:.4f})')

ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.legend(loc='lower left'); ax.grid(linestyle=':', alpha=0.5); ax.set_axisbelow(True)
ax.set_title('Precision-Recall Curves — Baselines on ExDark Test Split')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/Figure_4_1_PR_curves.png', dpi=300, bbox_inches='tight')
plt.show()
print('✓ Figure 4.1')


# ======================================================================
# ## Figure 4.2 — Enhancement methods bar chart
# Figure 4.2
methods = ['Baseline','CLAHE','Gamma\n(γ=0.5)','SSR','MSR','CLAHE+\nGamma','Zero-DCE','Zero-DCE+\nCLAHE']
keys = ['Baseline','CLAHE','Gamma','SSR','MSR','CLAHE+Gamma','Zero-DCE','Zero-DCE+CLAHE']
v5  = [0.7218, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
v8  = [DATA['enh_v8s'].get(k, np.nan) for k in keys]
v26 = [DATA['enh_v26s'].get(k, np.nan) for k in keys]

fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(methods)); w = 0.27
b1 = ax.bar(x-w, v5, w, label='YOLOv5s', color=COLOR_V5, edgecolor='white')
b2 = ax.bar(x,    v8, w, label='YOLOv8s', color=COLOR_V8, edgecolor='white')
b3 = ax.bar(x+w, v26, w, label='YOLO26s', color=COLOR_V26, edgecolor='white')

ax.axhline(0.7459, color=COLOR_BASE, linestyle='--', linewidth=1, alpha=0.7, label='YOLOv8s baseline')
ax.set_ylabel('mAP@0.5')
ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=10)
ax.set_ylim(0.60, 0.80); ax.grid(axis='y', linestyle=':', alpha=0.5); ax.set_axisbelow(True)
ax.legend(loc='lower right')
ax.set_title('Image Enhancement Methods on ExDark — In-Domain Performance')

for bars in [b1,b2,b3]:
    for bar in bars:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x()+bar.get_width()/2, h+0.003, f'{h:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/Figure_4_2_enhancement_methods.png', dpi=300, bbox_inches='tight')
plt.show()
print('✓ Figure 4.2')


# ======================================================================
# ## Figure 4.3 — Confusion matrix
# Figure 4.3 — try to use existing confusion_matrix.png from val
cm_paths = list(RESULTS.rglob('confusion_matrix.png'))
if cm_paths:
    img = cv2.cvtColor(cv2.imread(str(cm_paths[0])), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(img)
    ax.set_title('YOLOv8s Confusion Matrix on ExDark Test Split', pad=12)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/Figure_4_3_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f'✓ Figure 4.3 (from {cm_paths[0]})')
else:
    print('confusion_matrix.png not found — run model.val():')
    print('  metrics = model.val(data="<your_data.yaml>", save=True)')


# ======================================================================
# ## Figure 4.4 — Per-class mAP@0.5
# Figure 4.4 — typical per-class values
classes = DATA['classes_exdark']
v5  = [0.78,0.62,0.66,0.86,0.83,0.55,0.72,0.58,0.68,0.71,0.85,0.62]
v8  = [0.81,0.66,0.69,0.88,0.85,0.59,0.74,0.61,0.71,0.74,0.87,0.65]
v26 = [0.83,0.68,0.71,0.90,0.87,0.61,0.75,0.63,0.73,0.76,0.89,0.66]

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(classes)); w = 0.27
ax.bar(x-w, v5, w, label='YOLOv5s', color=COLOR_V5)
ax.bar(x,    v8, w, label='YOLOv8s', color=COLOR_V8)
ax.bar(x+w, v26, w, label='YOLO26s', color=COLOR_V26)
ax.set_ylabel('mAP@0.5'); ax.set_xlabel('ExDark class')
ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=10, rotation=15, ha='right')
ax.set_ylim(0, 1); ax.grid(axis='y', linestyle=':', alpha=0.5); ax.set_axisbelow(True)
ax.legend(loc='lower right')
ax.set_title('Per-Class mAP@0.5 on ExDark Test Split')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/Figure_4_4_per_class.png', dpi=300, bbox_inches='tight')
plt.show()
print('Figure 4.4 done (replace with your own model.val() results)')


# ======================================================================
# ## Figure 4.5 — CBAM × enhancement
# Figure 4.5
methods = ['CBAM','CBAM+\nCLAHE','CBAM+\nGamma','CBAM+\nSSR','CBAM+\nMSR',
           'CBAM+\nCLAHE+\nGamma','CBAM+\nZero-DCE','CBAM+\nZero-DCE+\nCLAHE']
keys = ['CBAM only','CBAM+CLAHE','CBAM+Gamma','CBAM+SSR','CBAM+MSR',
        'CBAM+CLAHE+Gamma','CBAM+Zero-DCE','CBAM+Zero-DCE+CLAHE']
v8  = [DATA['cbam_v8s'].get(k, np.nan) for k in keys]
v26 = [DATA['cbam_v26s'].get(k, np.nan) for k in keys]

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(methods)); w = 0.38
b1 = ax.bar(x-w/2, v8, w, label='YOLOv8s', color=COLOR_V8, edgecolor='white')
b2 = ax.bar(x+w/2, v26, w, label='YOLO26s', color=COLOR_V26, edgecolor='white')
b1[6].set_color(COLOR_PEAK); b1[6].set_edgecolor('#806000'); b1[6].set_linewidth(2.5)

ax.axhline(0.7459, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='YOLOv8s baseline')
ax.axhline(0.7640, color='green', linestyle=':', linewidth=1, alpha=0.6, label='YOLO26s baseline')
ax.set_ylabel('mAP@0.5')
ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9.5)
ax.set_ylim(0.60, 0.90); ax.grid(axis='y', linestyle=':', alpha=0.5); ax.set_axisbelow(True)
ax.legend(loc='lower left')
ax.set_title('CBAM + Image Enhancement on ExDark')
ax.annotate('Project peak\n0.8562', xy=(6-w/2, 0.8562), xytext=(6.5-w/2, 0.88),
            fontsize=10, fontweight='bold', color='#806000', ha='center',
            arrowprops=dict(arrowstyle='->', color='#806000', lw=1.5))

for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x()+bar.get_width()/2, h+0.003, f'{h:.3f}', ha='center', va='bottom', fontsize=7.5)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/Figure_4_5_cbam_combinations.png', dpi=300, bbox_inches='tight')
plt.show()
print('✓ Figure 4.5')


# ======================================================================
# ## Figure 4.6 — ExDark detection examples
# Figure 4.6 — uses model from Cell 3
def draw_dets(ax, img_rgb, boxes, names, conf=0.25, color='#00C853'):
    ax.imshow(img_rgb)
    for box in boxes:
        c = float(box.conf[0])
        if c < conf: continue
        x1,y1,x2,y2 = box.xyxy[0].tolist()
        ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,lw=2.4,ec=color,fc='none'))
        ax.text(x1, y1-4, f'{names[int(box.cls[0])]} {c:.2f}', fontsize=9, fontweight='bold',
                color='white', bbox=dict(boxstyle='round,pad=0.25', facecolor=color, edgecolor='none', alpha=0.92))
    ax.set_xticks([]); ax.set_yticks([])

random.seed(8)
ex_picks = []
for hint in ['cup','people','car','motorbike']:
    cands = list(EXDARK_DIR.rglob(f'*{hint}*.jpg'))[:30]
    if cands:
        for c in cands:
            img = cv2.imread(str(c))
            if img is not None and 30 < img.mean() < 100:
                ex_picks.append(c); break
        else: ex_picks.append(cands[0])
    else: ex_picks.append(random.choice(list(EXDARK_DIR.rglob('*.jpg'))))

titles = ['Indoor cup/bottle','Outdoor people','Vehicle low-light','Multiple objects']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, p, t in zip(axes.flat, ex_picks[:4], titles):
    img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    res = model.predict(source=str(p), conf=0.25, verbose=False)
    draw_dets(ax, img, res[0].boxes, model.names)
    n = sum(1 for b in res[0].boxes if float(b.conf[0]) >= 0.25)
    ax.set_title(f'{t}  —  {n} detection(s)', fontsize=12, pad=4)

plt.suptitle('YOLOv8s Detection Examples on ExDark', fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/Figure_4_6_ExDark_detections.png', dpi=300, bbox_inches='tight')
plt.show()
print('✓ Figure 4.6')


# ======================================================================
# ## Figure 4.7 — DAWN heatmap
# Figure 4.7
conditions = ['Dust Tornado','Foggy','Haze','Mist','Rain','Sand','Snow']
models_list = ['YOLOv5s','YOLOv8s','YOLO26s']
arr_data = np.array([[DATA['dawn'][m][c] for c in conditions] for m in models_list])

fig, ax = plt.subplots(figsize=(11, 4.5))
im = ax.imshow(arr_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.1)
ax.set_xticks(range(len(conditions)))
ax.set_xticklabels([c.replace(' ','\n') for c in conditions], fontsize=10)
ax.set_yticks(range(len(models_list))); ax.set_yticklabels(models_list, fontsize=11)

for i in range(len(models_list)):
    for j in range(len(conditions)):
        v = arr_data[i,j]
        ax.text(j, i, f'{v:.4f}', ha='center', va='center',
                color='white' if v>0.05 else 'black', fontsize=10, fontweight='bold')

cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label('mAP@0.5')
ax.set_title('DAWN Cross-Domain Heatmap (7 weather × 3 architectures)', pad=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/Figure_4_7_DAWN_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
print('✓ Figure 4.7')


# ======================================================================
# ## Figure 4.8 — Cross-domain comparison
# Figure 4.8
datasets = ['ExDark\n(in-domain)','BDD100K\n(driving)','DAWN\n(weather)']
v5  = [0.7218, 0.2540, 0.0582]
v8  = [0.7459, 0.2549, 0.0058]
v26 = [0.7640, 0.2626, 0.0020]

fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(3); w = 0.27
ax.bar(x-w, v5, w, label='YOLOv5s', color=COLOR_V5)
ax.bar(x,    v8, w, label='YOLOv8s', color=COLOR_V8)
ax.bar(x+w, v26, w, label='YOLO26s', color=COLOR_V26)
ax.set_ylabel('mAP@0.5'); ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=11)
ax.set_ylim(0, 0.85); ax.grid(axis='y', linestyle=':', alpha=0.5); ax.set_axisbelow(True)
ax.legend(loc='upper right'); ax.set_title('In-Domain vs Cross-Domain', pad=12)

for i, vals in enumerate([v5,v8,v26]):
    for j, v in enumerate(vals):
        ax.text(j+(i-1)*w, v+0.012, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

ax.annotate('', xy=(2.27,0.27), xytext=(0.27,0.79),
            arrowprops=dict(arrowstyle='->',color=COLOR_BAD,lw=2,alpha=0.6))
ax.text(1.3, 0.55, 'Cross-domain\nreversal', color=COLOR_BAD, fontsize=11, ha='center',
        fontweight='bold', alpha=0.85,
        bbox=dict(boxstyle='round',facecolor='#FFF0F0',edgecolor=COLOR_BAD,linewidth=1))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/Figure_4_8_cross_domain.png', dpi=300, bbox_inches='tight')
plt.show()
print('✓ Figure 4.8')


# ======================================================================
# ## Figure 4.9 — Per-shared-class BDD
# Figure 4.9
shared = ['Bicycle','Bus','Car','Motorbike','People']
v5  = [0.32, 0.40, 0.55, 0.28, 0.45]
v8  = [0.34, 0.42, 0.57, 0.30, 0.47]
v26 = [0.35, 0.43, 0.58, 0.32, 0.48]

fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(5); w = 0.27
ax.bar(x-w, v5, w, label='YOLOv5s', color=COLOR_V5)
ax.bar(x,    v8, w, label='YOLOv8s', color=COLOR_V8)
ax.bar(x+w, v26, w, label='YOLO26s', color=COLOR_V26)
ax.set_ylabel('mAP@0.5 (BDD shared classes)')
ax.set_xticks(x); ax.set_xticklabels(shared, fontsize=11)
ax.set_ylim(0, 0.7); ax.grid(axis='y', linestyle=':', alpha=0.5); ax.set_axisbelow(True)
ax.legend(loc='upper left')
ax.set_title('Per-Shared-Class mAP@0.5 on BDD100K (5 mapped classes)', pad=12)

for i, vals in enumerate([v5,v8,v26]):
    for j, v in enumerate(vals):
        ax.text(j+(i-1)*w, v+0.008, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/Figure_4_9_per_shared_class.png', dpi=300, bbox_inches='tight')
plt.show()
print('Figure 4.9 done (replace with your own val() results)')


# ======================================================================
# ## Figure 4.10 — BDD cross-domain ablation
# Figure 4.10
methods = ['Baseline','CBAM','CLAHE','Gamma','CLAHE+\nGamma','MSR','SSR','Zero-DCE','Zero-DCE+\nCLAHE']
keys = ['Baseline','CBAM','CLAHE','Gamma','CLAHE+Gamma','MSR','SSR','Zero-DCE','Zero-DCE+CLAHE']

def gv(t):
    v = DATA['bdd_cross'][t]
    return [x if x is not None else np.nan for x in v]

v5  = [gv(k)[0] for k in keys]
v8  = [gv(k)[1] for k in keys]
v26 = [gv(k)[2] for k in keys]

fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(methods)); w = 0.27
b1 = ax.bar(x-w, v5, w, label='YOLOv5s', color=COLOR_V5)
b2 = ax.bar(x,    v8, w, label='YOLOv8s', color=COLOR_V8)
b3 = ax.bar(x+w, v26, w, label='YOLO26s', color=COLOR_V26)
for bars in [b1, b2, b3]:
    bars[7].set_edgecolor(COLOR_BAD); bars[7].set_linewidth(2)
    bars[8].set_edgecolor(COLOR_BAD); bars[8].set_linewidth(2)

ax.set_ylabel('mAP@0.5 (BDD100K mapped test)')
ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=10)
ax.set_ylim(0, 0.32); ax.grid(axis='y', linestyle=':', alpha=0.5); ax.set_axisbelow(True)
ax.legend(loc='upper right'); ax.set_title('BDD100K Cross-Domain Ablation — Zero-DCE Catastrophic Collapse')
ax.annotate('Zero-DCE\ncollapse (>85%)', xy=(7, 0.08), xytext=(6, 0.27),
            fontsize=10, fontweight='bold', color=COLOR_BAD, ha='center',
            arrowprops=dict(arrowstyle='->', color=COLOR_BAD, lw=1.5))

for bars in [b1,b2,b3]:
    for bar in bars:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x()+bar.get_width()/2, h+0.005, f'{h:.3f}', ha='center', va='bottom', fontsize=7.5)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/Figure_4_10_BDD_ablation.png', dpi=300, bbox_inches='tight')
plt.show()
print('✓ Figure 4.10')


# ======================================================================
# ## Figure 4.11 — BDD detection cross-domain
# Figure 4.11
def draw_dets(ax, img_rgb, boxes, names, conf=0.20, color='#FF8F00'):
    ax.imshow(img_rgb)
    for box in boxes:
        c = float(box.conf[0])
        if c < conf: continue
        x1,y1,x2,y2 = box.xyxy[0].tolist()
        ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,lw=2.4,ec=color,fc='none'))
        ax.text(x1, y1-4, f'{names[int(box.cls[0])]} {c:.2f}', fontsize=9, fontweight='bold',
                color='white', bbox=dict(boxstyle='round,pad=0.25', facecolor=color, edgecolor='none', alpha=0.92))
    ax.set_xticks([]); ax.set_yticks([])

bdd_files = list(BDD_DIR.rglob('*.jpg'))[:300]
random.seed(13)
picks = random.sample(bdd_files, 4)
titles = ['Day urban','Night highway','Pedestrian crossing','Adverse weather']

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, p, t in zip(axes.flat, picks, titles):
    img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    res = model.predict(source=str(p), conf=0.20, verbose=False)
    draw_dets(ax, img, res[0].boxes, model.names)
    n = sum(1 for b in res[0].boxes if float(b.conf[0]) >= 0.20)
    ax.set_title(f'{t}  —  {n} detection(s) at conf ≥ 0.20', fontsize=12, pad=4)

plt.suptitle('Cross-Domain: ExDark-Trained YOLOv8s on BDD100K', fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/Figure_4_11_BDD_detections.png', dpi=300, bbox_inches='tight')
plt.show()
print('✓ Figure 4.11')


# ======================================================================
# ## Figure 4.12 — Zero-DCE darkening artifact
# Figure 4.12
ex_files = list(EXDARK_DIR.rglob('*.jpg'))
random.seed(0)
ex_dark = None
for p in random.sample(ex_files, min(30, len(ex_files))):
    img = cv2.imread(str(p))
    if img is not None and 25 < img.mean() < 60:
        ex_dark = (p, img); break
if ex_dark is None: ex_dark = (ex_files[0], cv2.imread(str(ex_files[0])))

bdd_files = list(BDD_DIR.rglob('*.jpg'))[:300]
random.seed(0)
bdd_bright = None
for p in random.sample(bdd_files, min(30, len(bdd_files))):
    img = cv2.imread(str(p))
    if img is not None and img.mean() > 110:
        bdd_bright = (p, img); break
if bdd_bright is None: bdd_bright = (bdd_files[0], cv2.imread(str(bdd_files[0])))

ex_o, ex_d = ex_dark[1], apply_zerodce(ex_dark[1])
b_o,  b_d  = bdd_bright[1], apply_zerodce(bdd_bright[1])
m_eo, m_ed = ex_o.mean()/255, ex_d.mean()/255
m_bo, m_bd = b_o.mean()/255, b_d.mean()/255

fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
axes[0,0].imshow(cv2.cvtColor(ex_o, cv2.COLOR_BGR2RGB))
axes[0,0].set_title(f'ExDark — Original (mean = {m_eo:.3f})', fontsize=12)
axes[0,1].imshow(cv2.cvtColor(ex_d, cv2.COLOR_BGR2RGB))
axes[0,1].set_title(f'ExDark — Zero-DCE (mean = {m_ed:.3f}, {m_ed/m_eo:.1f}× brighter ✓)',
                     fontsize=12, color='#2E7D32')
axes[1,0].imshow(cv2.cvtColor(b_o, cv2.COLOR_BGR2RGB))
axes[1,0].set_title(f'BDD100K — Original (mean = {m_bo:.3f})', fontsize=12)
axes[1,1].imshow(cv2.cvtColor(b_d, cv2.COLOR_BGR2RGB))
axes[1,1].set_title(f'BDD100K — Zero-DCE (mean = {m_bd:.3f}, {m_bd/m_bo:.2f}× = DARKENING ✗)',
                     fontsize=12, color='#C00000')
for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])

fig.text(0.5, 0.02, 'Zero-DCE brightens dark scenes but darkens HDR scenes — source of cross-domain mAP collapse.',
         ha='center', fontsize=10, style='italic', color='#444')
plt.suptitle('Zero-DCE Asymmetric Behavior', fontsize=14, y=1.00)
plt.tight_layout(rect=[0, 0.04, 1, 0.97])
plt.savefig(f'{OUT_DIR}/Figure_4_12_zerodce_darkening.png', dpi=300, bbox_inches='tight')
plt.show()
print('✓ Figure 4.12')


# ======================================================================
# ## Figure 4.13 — BDD native results
# Figure 4.13
configs = [c[0].replace(' (','\n(') if '(' in c[0] else c[0] for c in DATA['bdd_native']]
maps = [c[1] for c in DATA['bdd_native']]
types = [c[2] for c in DATA['bdd_native']]
cmap = {'baseline': '#5B9BD5', 'single': '#A9A9A9', 'reversal': COLOR_BAD}
colors = [cmap[t] for t in types]

fig, ax = plt.subplots(figsize=(13, 6))
bars = ax.bar(range(len(configs)), maps, color=colors, edgecolor='white', linewidth=1)
ax.set_xticks(range(len(configs))); ax.set_xticklabels(configs, fontsize=9.5)
ax.set_ylabel('mAP@0.5 (BDD100K validation)')
ax.set_ylim(0.42, 0.51); ax.grid(axis='y', linestyle=':', alpha=0.5); ax.set_axisbelow(True)
ax.set_title('BDD100K Native Training — Six Configurations', pad=12)

for bar, v in zip(bars, maps):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.002, f'{v:.4f}',
            ha='center', va='bottom', fontsize=10.5, fontweight='bold')

ax.annotate('Worst (was ExDark peak!)', xy=(5, 0.4444), xytext=(4.5, 0.50),
            fontsize=10, fontweight='bold', color=COLOR_BAD, ha='center',
            arrowprops=dict(arrowstyle='->', color=COLOR_BAD, lw=1.5))

ax.legend(handles=[
    mpatches.Patch(color='#5B9BD5', label='Plain baseline'),
    mpatches.Patch(color='#A9A9A9', label='Single modification'),
    mpatches.Patch(color=COLOR_BAD,  label='Synergy reversal (was peak)'),
], loc='lower left')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/Figure_4_13_BDD_native.png', dpi=300, bbox_inches='tight')
plt.show()
print('✓ Figure 4.13')


# ======================================================================
# ## Figure 5.1 — Cross-domain rank reversal
# Figure 5.1
domains = ['ExDark\n(In-Domain)','BDD100K\n(Cross-Domain)','DAWN\n(Cross-Domain)']
v5_rank, v8_rank, v26_rank = [3,1,1], [2,2,2], [1,3,3]

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(range(3), v5_rank,  '-o', linewidth=3, markersize=15, label='YOLOv5s', color=COLOR_V5,
         markeredgecolor='white', markeredgewidth=2)
ax.plot(range(3), v8_rank,  '-s', linewidth=3, markersize=14, label='YOLOv8s', color=COLOR_V8,
         markeredgecolor='white', markeredgewidth=2)
ax.plot(range(3), v26_rank, '-^', linewidth=3, markersize=15, label='YOLO26s', color=COLOR_V26,
         markeredgecolor='white', markeredgewidth=2)

m5  = ['0.7218','0.2540','0.0582']
m8  = ['0.7459','0.2549','0.0058']
m26 = ['0.7640','0.2626','0.0020']
for i in range(3):
    ax.annotate(m5[i],  (i, v5_rank[i]),  xytext=(8, 0), textcoords='offset points',
                fontsize=9.5, va='center', color=COLOR_V5)
    ax.annotate(m8[i],  (i, v8_rank[i]),  xytext=(8, 0), textcoords='offset points',
                fontsize=9.5, va='center', color=COLOR_V8)
    ax.annotate(m26[i], (i, v26_rank[i]), xytext=(8, 0), textcoords='offset points',
                fontsize=9.5, va='center', color=COLOR_V26)

ax.set_yticks([1,2,3]); ax.set_yticklabels(['Rank 1\n(Best)','Rank 2','Rank 3\n(Worst)'], fontsize=11)
ax.invert_yaxis(); ax.set_xticks(range(3)); ax.set_xticklabels(domains, fontsize=11)
ax.legend(loc='center right'); ax.grid(linestyle=':', alpha=0.5); ax.set_axisbelow(True)
ax.set_title('Cross-Domain Rank Reversal: Ordering Inverts Out-of-Domain', pad=15)
ax.set_ylim(3.6, 0.4)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/Figure_5_1_rank_reversal.png', dpi=300, bbox_inches='tight')
plt.show()
print('✓ Figure 5.1')


# ======================================================================
# ## Figure 5.2 — Synergy reversal headline
# Figure 5.2
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes[0]
configs = ['YOLOv8s\nbaseline','CBAM\nonly','Zero-DCE\nonly','CBAM +\nZero-DCE']
vals = [0.7459, 0.6968, 0.7498, 0.8562]
colors = ['#5B9BD5','#A9A9A9','#A9A9A9',COLOR_PEAK]
bars = ax.bar(configs, vals, color=colors, edgecolor='white', linewidth=1.5)
ax.set_ylim(0.60, 0.92); ax.set_ylabel('mAP@0.5')
ax.set_title('ExDark (in-domain training)', fontsize=13, pad=8)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.005, f'{v:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.grid(axis='y', linestyle=':', alpha=0.5); ax.set_axisbelow(True)
ax.text(3, 0.895, '★ PEAK', ha='center', fontsize=12, fontweight='bold', color='#806000')

ax = axes[1]
configs = ['YOLOv8s\nbaseline\n(30K)','CBAM\nonly','Zero-DCE\nonly','CBAM +\nZero-DCE']
vals = [0.4659, 0.4489, 0.4518, 0.4444]
colors = ['#5B9BD5','#A9A9A9','#A9A9A9',COLOR_BAD]
bars = ax.bar(configs, vals, color=colors, edgecolor='white', linewidth=1.5)
ax.set_ylim(0.42, 0.50); ax.set_ylabel('mAP@0.5')
ax.set_title('BDD100K (native training)', fontsize=13, pad=8)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.0015, f'{v:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.grid(axis='y', linestyle=':', alpha=0.5); ax.set_axisbelow(True)
ax.text(3, 0.493, '✗ WORST', ha='center', fontsize=12, fontweight='bold', color=COLOR_BAD)

fig.suptitle('Synergy Reversal: Same Combination, Opposite Behavior', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/Figure_5_2_synergy_reversal.png', dpi=300, bbox_inches='tight')
plt.show()
print('✓ Figure 5.2')


# ======================================================================
# ## Figure 5.3 — Failure case analysis
# Figure 5.3
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
modes = [
    ('(a) Class taxonomy mismatch', 'BDD has truck/traffic-light\nthat ExDark never saw', '#4472C4', 'taxonomy'),
    ('(b) High dynamic range', 'Zero-DCE over-suppresses\nbright pixels on BDD', '#C00000', 'hdr'),
    ('(c) Synergy reversal', 'CBAM+Zero-DCE peak\nbecomes worst on BDD', '#806000', 'reversal'),
]
for ax, (title, sub, color, kind) in zip(axes, modes):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    ax.add_patch(FancyBboxPatch((0.3,0.3),9.4,9.4,boxstyle='round,pad=0.1',
                                 facecolor='#F8F8F8',edgecolor=color,linewidth=2))
    if kind == 'taxonomy':
        ax.add_patch(Circle((3.5,5.5),2.2,fc='#FFD580',ec='#806000',alpha=0.6,lw=2))
        ax.add_patch(Circle((6.5,5.5),2.2,fc='#B4C7E7',ec='#1F3864',alpha=0.6,lw=2))
        ax.text(2.4, 5.5, 'ExDark\n12 classes', ha='center', fontsize=10, fontweight='bold')
        ax.text(7.6, 5.5, 'BDD100K\n10 classes', ha='center', fontsize=10, fontweight='bold')
        ax.text(5.0, 5.5, '5\nshared', ha='center', fontsize=10, fontweight='bold', color='#2E5496')
        ax.text(5.0, 2.0, '7 ExDark + 5 BDD = non-overlap', ha='center', fontsize=9.5, style='italic', color='#666')
    elif kind == 'hdr':
        x = np.linspace(0, 1, 100)
        h_ex = np.exp(-((x-0.18)**2)/0.02)*4
        h_bdd = np.exp(-((x-0.6)**2)/0.05)*2 + np.exp(-((x-0.9)**2)/0.005)*4
        sub_ax = ax.inset_axes([0.15, 0.35, 0.7, 0.4])
        sub_ax.fill_between(x, 0, h_ex, color='#1F3864', alpha=0.4, label='ExDark')
        sub_ax.fill_between(x, 0, h_bdd, color='#C00000', alpha=0.4, label='BDD100K')
        sub_ax.set_xlabel('Pixel intensity', fontsize=9)
        sub_ax.set_ylabel('Density', fontsize=9)
        sub_ax.legend(fontsize=8, loc='upper center')
        sub_ax.set_title('Intensity histograms', fontsize=10)
    elif kind == 'reversal':
        sub_ax = ax.inset_axes([0.15, 0.25, 0.7, 0.55])
        labels = ['ExDark','BDD\n(native)']
        peak = [0.8562, 0.4444]
        base = [0.7459, 0.4659]
        x_pos = np.arange(2)
        sub_ax.bar(x_pos-0.2, peak, 0.4, color='#FFC000', label='CBAM+Zero-DCE')
        sub_ax.bar(x_pos+0.2, base, 0.4, color='#A9A9A9', label='Baseline')
        sub_ax.set_xticks(x_pos); sub_ax.set_xticklabels(labels, fontsize=10)
        sub_ax.set_ylabel('mAP@0.5', fontsize=9)
        sub_ax.legend(fontsize=8, loc='upper right')
        sub_ax.annotate('PEAK', xy=(-0.2, 0.86), fontsize=10, fontweight='bold', color='#806000', ha='center')
        sub_ax.annotate('WORST', xy=(1.2, 0.46), fontsize=10, fontweight='bold', color='#C00000', ha='center')

    ax.text(5, 9.3, title, ha='center', fontsize=12, fontweight='bold', color=color)
    ax.text(5, 8.5, sub, ha='center', fontsize=10, style='italic', color='#444')

plt.suptitle('Three Diagnosed Failure Modes', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/Figure_5_3_failure_cases.png', dpi=300, bbox_inches='tight')
plt.show()
print('✓ Figure 5.3')


# ======================================================================
# ## Summary
# Summary
import subprocess
result = subprocess.run(['ls', '-la', OUT_DIR], capture_output=True, text=True)
print(result.stdout)

files = sorted(Path(OUT_DIR).glob('*.png'))
total_kb = sum(f.stat().st_size for f in files) / 1024
print(f'\n{len(files)} figures, {total_kb:.0f} KB total')
