"""Classical low-light image enhancement methods: CLAHE, gamma correction,
single-scale Retinex (SSR), and multi-scale Retinex (MSR)."""

import cv2
import numpy as np


def apply_clahe(bgr, clip_limit=3.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def apply_gamma(bgr, gamma=0.5):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(bgr, table)


def single_scale_retinex(bgr, sigma=80):
    img = bgr.astype(np.float32) + 1.0
    retinex = np.zeros_like(img, dtype=np.float32)
    for c in range(3):
        channel = img[:, :, c]
        blur = cv2.GaussianBlur(channel, (0, 0), sigma)
        retinex[:, :, c] = np.log(channel) - np.log(blur + 1.0)
    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
    return np.clip(retinex, 0, 255).astype(np.uint8)


def multi_scale_retinex(bgr, sigmas=(15, 80, 150)):
    img = bgr.astype(np.float32) + 1.0
    msr = np.zeros_like(img, dtype=np.float32)
    for sigma in sigmas:
        ret = np.zeros_like(img, dtype=np.float32)
        for c in range(3):
            channel = img[:, :, c]
            blur = cv2.GaussianBlur(channel, (0, 0), sigma)
            ret[:, :, c] = np.log(channel) - np.log(blur + 1.0)
        msr += ret
    msr /= len(sigmas)
    msr = cv2.normalize(msr, None, 0, 255, cv2.NORM_MINMAX)
    return np.clip(msr, 0, 255).astype(np.uint8)


def apply_clahe_gamma(bgr, gamma=0.5):
    return apply_gamma(apply_clahe(bgr), gamma=gamma)
