"""Trains a YOLO model (optionally with CBAM attention) on an enhanced
dataset variant. Skips experiments that already finished, and supports
resuming from a partially-completed run.

Usage:
    python train.py --data path/to/data.yaml --name yolov8s_cbam_clahe \\
        --seed path/to/seed_weights.pt --mode finetune --with-cbam
"""

import argparse
import os
import time

import pandas as pd
import torch
from ultralytics import YOLO

from models.cbam import add_cbam_to_backbone


def already_completed(results_dir, min_epoch=49):
    csv_path = f"{results_dir}/results.csv"
    if not os.path.exists(csv_path):
        return False
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        last_epoch = int(df.iloc[-1]["epoch"]) if len(df) > 0 else 0
        return last_epoch >= min_epoch
    except Exception:
        return False


def train_experiment(data_yaml, exp_name, results_dir, seed_weights="yolov8s.pt",
                      mode="scratch", epochs=50, imgsz=640, batch=16,
                      patience=20, with_cbam=False):
    exp_dir = f"{results_dir}/{exp_name}"

    if already_completed(exp_dir, min_epoch=epochs - 1):
        print(f"Skipping {exp_name}: already completed")
        return

    if not os.path.exists(data_yaml):
        print(f"Skipping {exp_name}: data yaml not found ({data_yaml})")
        return
    if mode != "scratch" and not os.path.exists(seed_weights):
        print(f"Skipping {exp_name}: seed weights not found ({seed_weights})")
        return

    print(f"\n{'=' * 60}\nTraining {exp_name} ({mode})\n{'=' * 60}")
    t_start = time.time()

    model = YOLO(seed_weights)
    if with_cbam:
        model = add_cbam_to_backbone(model)

    train_kwargs = dict(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=exp_name,
        project=results_dir,
        exist_ok=True,
        patience=patience,
        device=0,
        verbose=True,
    )
    if mode == "resume":
        train_kwargs["resume"] = True

    try:
        model.train(**train_kwargs)
    finally:
        del model
        torch.cuda.empty_cache()

    elapsed = (time.time() - t_start) / 60
    print(f"Finished {exp_name} in {elapsed:.1f} min")

    csv_path = f"{exp_dir}/results.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        if "metrics/mAP50(B)" in df.columns:
            print(f"Best mAP@0.5: {df['metrics/mAP50(B)'].max():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--name", required=True, help="Experiment name")
    parser.add_argument("--results-dir", default="/content/drive/MyDrive/Bitirme/results")
    parser.add_argument("--seed", default="yolov8s.pt", help="Seed/pretrained weights path")
    parser.add_argument("--mode", choices=["scratch", "finetune", "resume"], default="scratch")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--with-cbam", action="store_true")
    args = parser.parse_args()

    train_experiment(
        data_yaml=args.data,
        exp_name=args.name,
        results_dir=args.results_dir,
        seed_weights=args.seed,
        mode=args.mode,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        with_cbam=args.with_cbam,
    )
