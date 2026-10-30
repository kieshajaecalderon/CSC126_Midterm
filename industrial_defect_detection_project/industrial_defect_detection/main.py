"""
Industrial Defect Detection System
CSC 126 Computer Vision Midterm Project

Simple classical OpenCV pipeline (single image, no reference needed):
1. Image operations : grayscale, CLAHE, Gaussian blur, background subtraction
2. Segmentation     : object mask + statistical threshold on local differences
3. Analysis         : morphology, contours, area filtering, bounding boxes

Note: results are SUSPECTED defects for human review, not confirmed defects.

Run:
    python main.py
    python main.py --image path\\to\\photo.jpg
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MAX_DIM = 800            # image is resized so settings behave the same on any photo
EDGE_MARGIN = 5          # pixels trimmed from object edges (avoids outline false alarms)
MAX_REGIONS = 12         # show only the biggest suspects
BOX_COLOR = (0, 140, 255)  # orange (BGR)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_image(path: str):
    """Load an image, shrink it if large. Returns (image, error_message)."""
    if not path:
        return None, "No image path was provided."
    p = Path(path).expanduser()
    if not p.exists():
        return None, f"File not found: {p}"
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return None, f"Unsupported image format: {p.suffix}"
    image = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None, "OpenCV could not read this image."

    h, w = image.shape[:2]
    scale = MAX_DIM / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    return image, None


def choose_image_file():
    """Open a file picker."""
    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        return filedialog.askopenfilename(
            title="Select the defect photo",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
                       ("All files", "*.*")],
        )
    except tk.TclError as exc:
        print(f"Could not open file picker: {exc}")
        return ""
    finally:
        if root is not None:
            try:
                root.destroy()
            except tk.TclError:
                pass


# ----------------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------------
def get_object_mask(gray: np.ndarray) -> np.ndarray:
    """
    Find the object so we only inspect its surface.
    If the photo has a bright (white) background, that background is removed.
    Otherwise the whole image is inspected.
    """
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])

    if np.median(border) > 225:  # white / light background
        mask = (gray < 238).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        mask = filled
    else:
        mask = np.full_like(gray, 255)

    # Trim edges so the outline of the object is not flagged as a defect
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (EDGE_MARGIN * 2 + 1, EDGE_MARGIN * 2 + 1)
    )
    return cv2.erode(mask, k)


def detect_defects(image: np.ndarray, sensitivity: int, min_area: int):
    """
    Returns: enhanced gray, binary mask, list of regions (x, y, w, h, area)
    sensitivity: 1 (strict, few results) .. 10 (loose, many results)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Image operations
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    smooth = cv2.GaussianBlur(enhanced, (5, 5), 0)
    background = cv2.GaussianBlur(smooth, (41, 41), 0)
    difference = cv2.absdiff(smooth, background)

    # Segmentation (only inside the object)
    object_mask = get_object_mask(gray)
    values = difference[object_mask > 0]
    if values.size == 0:
        return enhanced, np.zeros_like(gray), []

    k = 4.0 - 0.3 * sensitivity                       # sensitivity 5 -> k = 2.5
    threshold = max(values.mean() + k * values.std(), 12)
    binary = ((difference > threshold) & (object_mask > 0)).astype(np.uint8) * 255

    # Clean up: remove specks, then merge nearby fragments into one region
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    )

    # Analysis
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        regions.append((x, y, w, h, area))

    regions.sort(key=lambda r: r[4], reverse=True)
    return enhanced, binary, regions[:MAX_REGIONS]


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------
def draw_results(image: np.ndarray, regions: list) -> np.ndarray:
    """Thin boxes + small numbers only."""
    result = image.copy()
    for i, (x, y, w, h, _) in enumerate(regions, start=1):
        pad = 4
        p1 = (max(0, x - pad), max(0, y - pad))
        p2 = (min(result.shape[1] - 1, x + w + pad), min(result.shape[0] - 1, y + h + pad))
        cv2.rectangle(result, p1, p2, BOX_COLOR, 2)
        cv2.putText(result, str(i), (p1[0], max(14, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, BOX_COLOR, 2, cv2.LINE_AA)
    return result


def titled(panel: np.ndarray, title: str) -> np.ndarray:
    if panel.ndim == 2:
        panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
    panel = panel.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 30), (35, 35, 35), -1)
    cv2.putText(panel, title, (10, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def build_dashboard(image, mask, annotated, count, sensitivity, min_area):
    """Three panels in one row: original | mask | result, plus a footer."""
    panels = [
        titled(image, "Original"),
        titled(mask, "Defect Mask"),
        titled(annotated, f"Suspected Defects: {count}"),
    ]
    row = np.hstack(panels)
    footer = np.zeros((36, row.shape[1], 3), dtype=np.uint8)
    text = (f"Sensitivity: {sensitivity}/10 | Min area: {min_area}px | "
            "O=open  S=save  +/-=sensitivity  [/]=min area  R=reset  Q=quit")
    cv2.putText(footer, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack((row, footer))


# ----------------------------------------------------------------------------
# Saving / reporting
# ----------------------------------------------------------------------------
def save_results(source_path, annotated, mask):
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(source_path).stem if source_path else "inspection"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    a_path = out / f"{stem}_result_{stamp}.png"
    m_path = out / f"{stem}_mask_{stamp}.png"
    if not cv2.imwrite(str(a_path), annotated):
        raise OSError(f"Could not save {a_path}")
    if not cv2.imwrite(str(m_path), mask):
        raise OSError(f"Could not save {m_path}")
    return a_path, m_path


def print_report(regions):
    print(f"\n--- Inspection report: {len(regions)} suspected region(s) ---")
    for i, (x, y, w, h, area) in enumerate(regions, start=1):
        print(f"{i}. area={area:.0f}px², box=(x:{x}, y:{y}, w:{w}, h:{h})")
    print("Reminder: these are candidates for human review, not confirmed defects.")


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def run_application(args):
    image_path = args.image or choose_image_file()
    image, error = load_image(image_path)
    if error:
        print(f"Image error: {error}")
        print("Use: python main.py --image path\\to\\image.jpg")
        return 1

    default_sens = int(np.clip(args.sensitivity, 1, 10))
    default_area = max(0, args.min_area)
    sensitivity, min_area = default_sens, default_area
    current_path = image_path

    window = "CSC 126 - Industrial Defect Detection"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1500, 560)

    print("Keys: O=open  S=save  +/-=sensitivity  [ / ]=min area  R=reset  Q/ESC=quit")

    needs_update = True
    mask = annotated = None
    regions = []

    while True:
        try:
            if needs_update:
                enhanced, mask, regions = detect_defects(image, sensitivity, min_area)
                annotated = draw_results(image, regions)
                dashboard = build_dashboard(
                    image, mask, annotated, len(regions), sensitivity, min_area
                )
                cv2.imshow(window, dashboard)
                needs_update = False

            key = cv2.waitKey(30) & 0xFF

            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == 255:
                continue

            if key in (ord("q"), 27):
                break
            elif key == ord("o"):
                path = choose_image_file()
                if path:
                    new_image, err = load_image(path)
                    if err:
                        print(f"Image error: {err}")
                    else:
                        image, current_path = new_image, path
                        print(f"Loaded: {path}")
                        needs_update = True
            elif key == ord("s"):
                try:
                    a_path, m_path = save_results(current_path, annotated, mask)
                    print(f"Saved: {a_path}\nSaved: {m_path}")
                    print_report(regions)
                except OSError as exc:
                    print(f"Save error: {exc}")
            elif key in (ord("+"), ord("=")):
                sensitivity = min(10, sensitivity + 1)
                needs_update = True
            elif key in (ord("-"), ord("_")):
                sensitivity = max(1, sensitivity - 1)
                needs_update = True
            elif key == ord("]"):
                min_area += 20
                needs_update = True
            elif key == ord("["):
                min_area = max(0, min_area - 20)
                needs_update = True
            elif key == ord("r"):
                sensitivity, min_area = default_sens, default_area
                needs_update = True

        except cv2.error as exc:
            print(f"OpenCV processing error: {exc}")
            break

    cv2.destroyAllWindows()
    return 0


def parse_arguments():
    parser = argparse.ArgumentParser(description="Simple OpenCV defect detection.")
    parser.add_argument("--image", type=str, default="",
                        help="Path to the defect photo. Opens a file picker if omitted.")
    parser.add_argument("--sensitivity", type=int, default=5,
                        help="1 (strict) to 10 (loose). Default 5.")
    parser.add_argument("--min-area", type=int, default=150,
                        help="Smallest region to report, in pixels². Default 150.")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        sys.exit(run_application(parse_arguments()))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        cv2.destroyAllWindows()
        sys.exit(0)