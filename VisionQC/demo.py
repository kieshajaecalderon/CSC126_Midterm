"""Build one side-by-side demo figure: Reference | Normal test | Defective test.

Rows: original image, segmentation mask, annotated result.
Run (one line):
  python demo.py --reference data/samples/reference/reference.png --normal data/samples/test/normal/normal_01.png --defect data/samples/test/defective/defect_scratch_01.png
"""
import argparse
import sys

import cv2
import numpy as np

from visionqc.config import load_params
from visionqc.inspector import inspect_pair
from visionqc.io_utils import load_image, save_image

CELL_W, CELL_H, BAR_H = 250, 250, 25


def to_bgr(image):
    """Masks are 1-channel; convert so every panel has 3 channels before stacking."""
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image


def make_cell(image, caption):
    """Letterbox an image into a fixed-size cell with a caption bar on top."""
    cell = np.zeros((CELL_H, CELL_W, 3), np.uint8)
    if image is not None:
        image = to_bgr(image)
        h, w = image.shape[:2]
        scale = min(CELL_W / w, (CELL_H - BAR_H) / h)
        size = (max(1, int(w * scale)), max(1, int(h * scale)))
        small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        y0 = BAR_H + (CELL_H - BAR_H - size[1]) // 2
        x0 = (CELL_W - size[0]) // 2
        cell[y0:y0 + size[1], x0:x0 + size[0]] = small
    cv2.rectangle(cell, (0, 0), (CELL_W, BAR_H), (40, 40, 40), -1)
    cv2.putText(cell, caption, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return cell


def result_caption(result):
    return f"{result.status} | {len(result.regions)} region(s)"


def main():
    parser = argparse.ArgumentParser(description="Reference / normal / defective figure.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--normal", required=True)
    parser.add_argument("--defect", required=True)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output", default="results/demo_figure.png")
    parser.add_argument("--show", action="store_true", help="also open a window")
    args = parser.parse_args()

    try:
        params = load_params(args.config)
        reference = load_image(args.reference)
        normal_result = inspect_pair(reference, load_image(args.normal), params)
        defect_result = inspect_pair(reference, load_image(args.defect), params)
    except ValueError as error:
        print(f"ERROR: {error}")
        sys.exit(1)

    ref_view = normal_result.reference_view
    top = [make_cell(ref_view, "Reference (normal product)"),
           make_cell(normal_result.test_view, "Test: expected normal"),
           make_cell(defect_result.test_view, "Test: expected defective")]
    middle = [make_cell(None, "Mask (reference has none)"),
              make_cell(normal_result.final_mask, "Segmentation mask"),
              make_cell(defect_result.final_mask, "Segmentation mask")]
    bottom = [make_cell(None, "Annotated (reference has none)"),
              make_cell(normal_result.annotated, result_caption(normal_result)),
              make_cell(defect_result.annotated, result_caption(defect_result))]

    figure = np.vstack([np.hstack(row) for row in (top, middle, bottom)])

    try:
        import os
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        save_image(args.output, figure)
    except (OSError, ValueError) as error:
        print(f"Could not save figure: {error}")
        sys.exit(1)
    print(f"Saved: {args.output}")

    if args.show:
        cv2.imshow("VisionQC demo (press any key)", figure)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()