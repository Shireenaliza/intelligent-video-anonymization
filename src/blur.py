import cv2
import numpy as np


def _odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 else value + 1


def gaussian_blur(face: np.ndarray, strength: int = 31) -> np.ndarray:
    k = _odd(strength)
    return cv2.GaussianBlur(face, (k, k), 0)


def mosaic_blur(face: np.ndarray, block_size: int = 12) -> np.ndarray:
    h, w = face.shape[:2]
    small_w = max(1, w // block_size)
    small_h = max(1, h // block_size)

    small = cv2.resize(
        face,
        (small_w, small_h),
        interpolation=cv2.INTER_LINEAR,
    )
    return cv2.resize(
        small,
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )


def dot_blur(face: np.ndarray, strength: int = 9) -> np.ndarray:
    # Circular/dot-shaped custom averaging kernel using OpenCV filtering.
    k = _odd(strength)
    kernel = np.zeros((k, k), dtype=np.float32)

    center = k // 2
    yy, xx = np.ogrid[:k, :k]
    radius = max(1, k // 3)
    mask = (xx - center) ** 2 + (yy - center) ** 2 <= radius ** 2

    kernel[mask] = 1.0 / np.count_nonzero(mask)
    return cv2.filter2D(face, -1, kernel)


def triangle_blur(face: np.ndarray, strength: int = 15) -> np.ndarray:
    # Triangular weighted kernel for a smooth custom blur.
    k = _odd(strength)
    kernel_1d = np.bartlett(k).astype(np.float32)

    if kernel_1d.sum() == 0:
        kernel_1d[:] = 1.0

    kernel_1d /= kernel_1d.sum()
    kernel = np.outer(kernel_1d, kernel_1d)

    return cv2.filter2D(face, -1, kernel)


def apply_blur(
    face: np.ndarray,
    method: str = "gaussian",
    strength: int = 31,
    mosaic_block_size: int = 12,
) -> np.ndarray:
    method = method.lower()

    if method == "gaussian":
        return gaussian_blur(face, strength)

    if method == "mosaic":
        return mosaic_blur(face, mosaic_block_size)

    if method == "dot":
        return dot_blur(face, strength)

    if method == "triangle":
        return triangle_blur(face, strength)

    raise ValueError(f"Unknown blur method: {method}")
