"""Runs every trained model/enhancement-method combination on a single image
and renders the detections of all of them side by side in one grid figure.

Usage:
    python visualize_comparison.py --image path/to/image.jpg --checkpoints-dir path/to/checkpoints
"""

import argparse
import math
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

from enhancement.classical import (
    apply_clahe,
    apply_clahe_gamma,
    apply_gamma,
    multi_scale_retinex,
    single_scale_retinex,
)
from enhancement.zero_dce import ZeroDCEEnhancer

IMG_SIZE = 640
CONF_THRES = 0.25
LINE_THICKNESS = 2
FONT_SCALE = 0.5

_zero_dce_enhancer = None


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_bgr(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def set_zero_dce_checkpoint(checkpoint_path, device="cuda"):
    global _zero_dce_enhancer
    _zero_dce_enhancer = ZeroDCEEnhancer(checkpoint_path, device=device)


def apply_zero_dce(bgr):
    if _zero_dce_enhancer is None:
        raise RuntimeError("Call set_zero_dce_checkpoint() before using the zero_dce method")
    return _zero_dce_enhancer.enhance(bgr)


def apply_zero_dce_clahe(bgr):
    return apply_clahe(apply_zero_dce(bgr))


def preprocess_image(bgr, method_name):
    method_name = method_name.lower()
    methods = {
        "baseline": lambda img: img.copy(),
        "clahe": apply_clahe,
        "gamma": lambda img: apply_gamma(img, gamma=0.5),
        "ssr": lambda img: single_scale_retinex(img, sigma=80),
        "msr": lambda img: multi_scale_retinex(img, sigmas=(15, 80, 150)),
        "zero_dce": apply_zero_dce,
        "clahe_gamma": apply_clahe_gamma,
        "zero_dce_clahe": apply_zero_dce_clahe,
    }
    if method_name not in methods:
        raise ValueError(f"Unknown method: {method_name}")
    return methods[method_name](bgr)


def draw_box(img, xyxy, label, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1)
    y_text = max(y1 - 5, th + 5)
    cv2.rectangle(img, (x1, y_text - th - 4), (x1 + tw + 4, y_text), color, -1)
    cv2.putText(img, label, (x1 + 2, y_text - 2),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), 1, cv2.LINE_AA)
    return img


_loaded_models = {}


def load_model_cached(model_type, weights_path, yolov5_repo=None):
    key = (model_type, weights_path)
    if key in _loaded_models:
        return _loaded_models[key]

    if model_type == "yolov5":
        model = torch.hub.load(yolov5_repo, "custom", path=weights_path, source="local")
        model.conf = CONF_THRES
        model.iou = 0.45
    elif model_type in ("yolov8", "yolo26", "ultralytics"):
        model = YOLO(weights_path)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    _loaded_models[key] = model
    return model


def run_yolov5(model, bgr_img):
    rgb_img = bgr_to_rgb(bgr_img)
    results = model(rgb_img, size=IMG_SIZE)
    df = results.pandas().xyxy[0]

    drawn = bgr_img.copy()
    for _, row in df.iterrows():
        xyxy = row[["xmin", "ymin", "xmax", "ymax"]].values
        label = f"{row['name']} {float(row['confidence']):.2f}"
        drawn = draw_box(drawn, xyxy, label, thickness=LINE_THICKNESS)
    return drawn


def run_ultralytics(model, bgr_img):
    rgb_img = bgr_to_rgb(bgr_img)
    results = model.predict(source=rgb_img, imgsz=IMG_SIZE, conf=CONF_THRES, verbose=False)
    res = results[0]
    drawn = bgr_img.copy()

    if res.boxes is not None and len(res.boxes) > 0:
        names = res.names
        boxes_xyxy = res.boxes.xyxy.cpu().numpy()
        boxes_conf = res.boxes.conf.cpu().numpy()
        boxes_cls = res.boxes.cls.cpu().numpy().astype(int)

        for xyxy, conf, cls_id in zip(boxes_xyxy, boxes_conf, boxes_cls):
            cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
            label = f"{cls_name} {conf:.2f}"
            drawn = draw_box(drawn, xyxy, label, thickness=LINE_THICKNESS)
    return drawn


def run_one_config(bgr_img, model_type, weights_path, preprocess_name, yolov5_repo=None):
    processed = preprocess_image(bgr_img, preprocess_name)
    model = load_model_cached(model_type, weights_path, yolov5_repo=yolov5_repo)
    if model_type == "yolov5":
        return run_yolov5(model, processed)
    return run_ultralytics(model, processed)


PREPROCESS_BY_KEY = {
    "baseline": "baseline", "clahe": "clahe", "gamma": "gamma",
    "ssr": "ssr", "msr": "msr", "zero_dce": "zero_dce",
    "clahe_gamma": "clahe_gamma", "zero_dce_clahe": "zero_dce_clahe",
    "cbam": "baseline", "cbam_clahe": "clahe", "cbam_gamma": "gamma",
    "cbam_ssr": "ssr", "cbam_msr": "msr", "cbam_zero_dce": "zero_dce",
    "cbam_clahe_gamma": "clahe_gamma", "cbam_zero_dce_clahe": "zero_dce_clahe",
}


def build_run_list(checkpoints):
    """checkpoints: {model_name: {"model_type": ..., method_key: weights_path or None, ...}}"""
    runs = []
    for model_name, info in checkpoints.items():
        model_type = info["model_type"]
        for key, ckpt in info.items():
            if key == "model_type" or ckpt is None:
                continue
            if not os.path.exists(ckpt):
                print(f"[SKIP] checkpoint not found: {ckpt}")
                continue
            runs.append({
                "title": f"{model_name}\n{key}",
                "model_type": model_type,
                "weights_path": ckpt,
                "preprocess_name": PREPROCESS_BY_KEY[key],
            })
    return runs


def render_grid(image_path, checkpoints, output_path, yolov5_repo=None, ncols=4):
    original_bgr = read_bgr(image_path)
    runs = build_run_list(checkpoints)
    print(f"Total runs found: {len(runs)}")

    all_outputs = [("Original", bgr_to_rgb(original_bgr))]
    for i, run in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {run['title']}")
        try:
            out_bgr = run_one_config(
                original_bgr, run["model_type"], run["weights_path"],
                run["preprocess_name"], yolov5_repo=yolov5_repo,
            )
            all_outputs.append((run["title"], bgr_to_rgb(out_bgr)))
        except Exception as e:
            print(f"ERROR in {run['title']}: {e}")
            err = np.ones_like(original_bgr) * 255
            cv2.putText(err, "ERROR", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.putText(err, run["title"], (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            all_outputs.append((run["title"], bgr_to_rgb(err)))

    n = len(all_outputs)
    nrows = math.ceil(n / ncols)
    plt.figure(figsize=(5 * ncols, 4.5 * nrows))
    for idx, (title, img_rgb) in enumerate(all_outputs, 1):
        plt.subplot(nrows, ncols, idx)
        plt.imshow(img_rgb)
        plt.title(title, fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoints-dir", required=True,
                         help="Directory containing per-model/method best.pt files, "
                              "e.g. <dir>/yolov8s_baseline/best.pt")
    parser.add_argument("--zero-dce-checkpoint", default=None)
    parser.add_argument("--yolov5-repo", default=None)
    parser.add_argument("--output", default="all_methods_all_models_grid.png")
    args = parser.parse_args()

    if args.zero_dce_checkpoint:
        set_zero_dce_checkpoint(args.zero_dce_checkpoint)

    ckpt_dir = Path(args.checkpoints_dir)
    method_keys = list(PREPROCESS_BY_KEY.keys())

    checkpoints = {
        "YOLOv5s": {"model_type": "yolov5", **{
            k: str(ckpt_dir / f"yolov5s_{k}" / "best.pt") for k in method_keys
        }},
        "YOLOv8s": {"model_type": "ultralytics", **{
            k: str(ckpt_dir / f"yolov8s_{k}" / "best.pt") for k in method_keys
        }},
        "YOLO26s": {"model_type": "ultralytics", **{
            k: str(ckpt_dir / f"yolo26s_{k}" / "best.pt") for k in method_keys
        }},
    }

    render_grid(args.image, checkpoints, args.output, yolov5_repo=args.yolov5_repo)
