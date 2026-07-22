"""Sanity check: counts raw ExDark images per class before conversion."""

import os

from exdark_to_yolo import CLASS_NAMES


def count_classes(base_path):
    total = 0
    for cls in CLASS_NAMES:
        cls_path = os.path.join(base_path, cls)
        if os.path.exists(cls_path):
            count = len(os.listdir(cls_path))
            total += count
            print(f"{cls}: {count} images")

    print(f"\nTotal: {total} images")
    print(f"Total classes: {len(CLASS_NAMES)}")


if __name__ == "__main__":
    count_classes("/content/exdark_raw/ExDark")
