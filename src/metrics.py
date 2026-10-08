from typing import Dict

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def mse(original: np.ndarray, processed: np.ndarray) -> float:
    original = original.astype(np.float32)
    processed = processed.astype(np.float32)

    return float(np.mean((original - processed) ** 2))


def psnr(original: np.ndarray, processed: np.ndarray) -> float:
    return float(
        peak_signal_noise_ratio(
            original,
            processed,
            data_range=255,
        )
    )


def ssim(original: np.ndarray, processed: np.ndarray) -> float:
    if original.ndim == 3:
        return float(
            structural_similarity(
                original,
                processed,
                channel_axis=2,
                data_range=255,
            )
        )

    return float(
        structural_similarity(
            original,
            processed,
            data_range=255,
        )
    )


def laplacian_variance(image: np.ndarray) -> float:
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3
        else image
    )

    return float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )


def tenengrad(image: np.ndarray) -> float:
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3
        else image
    )

    gx = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3,
    )

    return float(np.mean(gx * gx + gy * gy))


def brenner(image: np.ndarray) -> float:
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3
        else image
    )

    diff = (
        gray[:, 2:].astype(np.float32)
        - gray[:, :-2].astype(np.float32)
    )

    return float(np.mean(diff * diff))


def evaluate_pair(
    original: np.ndarray,
    processed: np.ndarray,
) -> Dict[str, float]:
    return {
        "mse": mse(original, processed),
        "psnr": psnr(original, processed),
        "ssim": ssim(original, processed),
        "laplacian_variance": laplacian_variance(processed),
        "tenengrad": tenengrad(processed),
        "brenner": brenner(processed),
    }
