"""TOPIC 1 - IMAGE OPERATIONS.

Pipeline for each image:
  resize -> grayscale -> Gaussian blur (noise) -> CLAHE (contrast)
Morphological operations are applied later, on the binary mask (segmentation.py).
"""
import cv2


def resize_to_width(image, width):
    """Resize keeping the aspect ratio so both images are comparable and fast to process."""
    height_old, width_old = image.shape[:2]
    if width_old == width:
        return image.copy()
    scale = width / width_old
    new_height = max(1, int(round(height_old * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, (width, new_height), interpolation=interpolation)


def to_grayscale(image):
    """cv2.cvtColor: color -> one brightness channel. Differences become one number per pixel."""
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def reduce_noise(gray, kernel_size):
    """cv2.GaussianBlur: smooths sensor noise so it is not mistaken for defects."""
    if kernel_size <= 1:
        return gray
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)


def enhance_contrast(gray, clip_limit):
    """cv2.createCLAHE: local contrast enhancement so faint defects become more visible."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(gray)


def preprocess(image_bgr, params):
    """Return the preprocessed grayscale image used for comparison."""
    gray = to_grayscale(image_bgr)
    smooth = reduce_noise(gray, params.blur_kernel)
    if params.use_clahe:
        return enhance_contrast(smooth, params.clahe_clip_limit)
    return smooth