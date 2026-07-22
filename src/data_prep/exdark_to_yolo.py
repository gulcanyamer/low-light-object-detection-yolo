"""Converts the ExDark dataset (raw images + bounding-box annotation files)
into YOLO-format train/val/test splits with a data.yaml file."""

import os
import shutil

import cv2
from sklearn.model_selection import train_test_split

CLASS_NAMES = [
    "Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat",
    "Chair", "Cup", "Dog", "Motorbike", "People", "Table",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def convert_annotation(anno_path, img_w, img_h):
    """Parses one ExDark annotation file and returns YOLO-format lines."""
    yolo_lines = []
    with open(anno_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            class_name = parts[0]
            if class_name not in CLASS_TO_ID:
                continue
            try:
                left, top, box_w, box_h = (int(float(p)) for p in parts[1:5])
            except ValueError:
                continue

            cx = max(0, min(1, (left + box_w / 2) / img_w))
            cy = max(0, min(1, (top + box_h / 2) / img_h))
            nw = max(0, min(1, box_w / img_w))
            nh = max(0, min(1, box_h / img_h))
            yolo_lines.append(f"{CLASS_TO_ID[class_name]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return yolo_lines


def build_dataset(images_dir, annotations_dir, output_dir, test_size=0.30, val_split=0.50, seed=42):
    for split in ["train", "val", "test"]:
        os.makedirs(f"{output_dir}/{split}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/{split}/labels", exist_ok=True)

    all_data = []
    for class_name in CLASS_NAMES:
        img_dir = f"{images_dir}/{class_name}"
        anno_dir = f"{annotations_dir}/{class_name}"
        for img_file in sorted(os.listdir(img_dir)):
            if img_file.startswith("."):
                continue
            img_path = f"{img_dir}/{img_file}"
            anno_path = f"{anno_dir}/{img_file}.txt"
            if not os.path.exists(anno_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            yolo_lines = convert_annotation(anno_path, w, h)
            if yolo_lines:
                all_data.append((img_path, yolo_lines))

    train_data, temp_data = train_test_split(all_data, test_size=test_size, random_state=seed)
    val_data, test_data = train_test_split(temp_data, test_size=val_split, random_state=seed)

    for data_list, split in [(train_data, "train"), (val_data, "val"), (test_data, "test")]:
        for img_path, yolo_lines in data_list:
            filename = os.path.basename(img_path)
            name_no_ext = os.path.splitext(filename)[0]
            dst_img = f"{output_dir}/{split}/images/{filename}"
            shutil.copy2(img_path, dst_img)
            with open(f"{output_dir}/{split}/labels/{name_no_ext}.txt", "w") as f:
                f.write("\n".join(yolo_lines))

    with open(f"{output_dir}/exdark.yaml", "w") as f:
        f.write(
            f"train: {output_dir}/train/images\n"
            f"val: {output_dir}/val/images\n"
            f"test: {output_dir}/test/images\n\n"
            f"nc: {len(CLASS_NAMES)}\n"
            f"names: {CLASS_NAMES}\n"
        )

    print(f"Done: {len(train_data)} train / {len(val_data)} val / {len(test_data)} test images")


if __name__ == "__main__":
    build_dataset(
        images_dir="/content/exdark_raw/ExDark",
        annotations_dir="/content/exdark_raw/ExDark_Annno",
        output_dir="/content/ExDark_YOLO",
    )
