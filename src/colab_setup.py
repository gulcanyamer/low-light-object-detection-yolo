"""Google Colab environment setup: installs Ultralytics, mounts Drive,
and creates the project folder structure used by the rest of the pipeline."""

import os

import torch
from google.colab import drive


def install_dependencies():
    os.system("pip install -q ultralytics")


def check_gpu():
    print("Ultralytics installed!")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")


def mount_drive():
    drive.mount("/content/drive")


def create_project_dirs(root="/content/drive/MyDrive/Bitirme"):
    os.makedirs(f"{root}/data", exist_ok=True)
    os.makedirs(f"{root}/results", exist_ok=True)
    print("Project folders created!")


if __name__ == "__main__":
    install_dependencies()
    check_gpu()
    mount_drive()
    create_project_dirs()
