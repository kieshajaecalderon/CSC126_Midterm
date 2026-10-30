"""Safe image loading/saving (works with Windows paths that contain non-ASCII characters)."""
import os
import cv2
import numpy as np

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def load_image(path):
    """Read an image as a 3-channel BGR array. Raises ValueError with a clear message."""
    if not path:
        raise ValueError("No file was selected.")
    if not os.path.isfile(path):
        raise ValueError(f"File not found: {path}")
    extension = os.path.splitext(path)[1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format '{extension}'. Use: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    try:
        raw = np.fromfile(path, dtype=np.uint8)
    except OSError as error:
        raise ValueError(f"Cannot read the file: {error}")
    if raw.size == 0:
        raise ValueError("The file is empty.")
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)  # always returns BGR (or None)
    if image is None or image.size == 0:
        raise ValueError("The file is not a valid or readable image.")
    return image


def save_image(path, image):
    """Write an image to disk. Raises ValueError if it cannot be saved."""
    extension = os.path.splitext(path)[1] or ".png"
    success, buffer = cv2.imencode(extension, image)
    if not success:
        raise ValueError(f"Could not encode image as '{extension}'.")
    try:
        buffer.tofile(path)
    except OSError as error:
        raise ValueError(f"Could not write '{path}': {error}")