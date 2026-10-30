"""Evaluate VisionQC on labeled images. Produces REAL numbers from YOUR data.

Example (synthetic samples):
  python evaluate.py --reference data/samples/reference/reference.png ^
      --normal-dirs data/samples/test/normal ^
      --defect-dirs data/samples/test/defective

Example (MVTec AD, one category, e.g. 'transistor'):
  python evaluate.py --reference data/mvtec_ad/transistor/train/good/000.png ^
      --normal-dirs data/mvtec_ad/transistor/test/good ^
      --defect-dirs data/mvtec_ad/transistor/test/bent_lead data/mvtec_ad/transistor/test/cut_lead

Positive class = defective. 'Manual Review Required' counts as FLAGGED (positive) by
default because a human must look at it; change with --manual-as normal.
"""
import argparse
import csv
import os
import sys

from visionqc.config import InspectionParams, load_params
from visionqc.inspector import inspect_pair, STATUS_DEFECT, STATUS_REVIEW
from visionqc.io_utils import load_image, SUPPORTED_EXTENSIONS


def list_images(folder):
    if not os.path.isdir(folder):
        print(f"WARNING: folder not found: {folder}")
        return []
    names = sorted(n for n in os.listdir(folder) if n.lower().endswith(SUPPORTED_EXTENSIONS))
    return [os.path.join(folder, n) for n in names]


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else None


def format_value(value):
    return "n/a" if value is None else f"{value:.3f}"


def compute_metrics(tp, fp, tn, fn):
    accuracy = safe_divide(tp + tn, tp + fp + tn + fn)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None
    return accuracy, precision, recall, f1


def run_once(reference, samples, params, manual_as_defect, csv_path):
    tp = fp = tn = fn = skipped = review = 0
    rows = []
    for path, actual in samples:
        try:
            result = inspect_pair(reference, load_image(path), params)
        except ValueError as error:
            print(f"  skipped {os.path.basename(path)}: {error}")
            skipped += 1
            continue
        if result.status == STATUS_REVIEW:
            review += 1
            predicted = 1 if manual_as_defect else 0
        else:
            predicted = 1 if result.status == STATUS_DEFECT else 0

        if actual == 1 and predicted == 1: tp += 1
        elif actual == 0 and predicted == 1: fp += 1
        elif actual == 0 and predicted == 0: tn += 1
        else: fn += 1
        rows.append([path, "defective" if actual else "normal", result.status,
                     len(result.regions), round(result.total_area, 1), round(result.coverage_percent, 4)])

    if csv_path:
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["file", "true_label", "predicted_status", "regions", "area_px", "coverage_percent"])
                writer.writerows(rows)
        except OSError as error:
            print(f"Could not write {csv_path}: {error}")

    accuracy, precision, recall, f1 = compute_metrics(tp, fp, tn, fn)
    total = tp + fp + tn + fn
    print(f"\nThreshold={params.threshold}  min_area={params.min_area}  (evaluated {total}, skipped {skipped}, "
          f"manual-review {review})")
    print("                 predicted defect | predicted normal")
    print(f"  actual defect  {tp:>16} | {fn:>15}   (TP | FN)")
    print(f"  actual normal  {fp:>16} | {tn:>15}   (FP | TN)")
    print(f"  Accuracy={format_value(accuracy)}  Precision={format_value(precision)}  "
          f"Recall={format_value(recall)}  F1={format_value(f1)}")
    return total


def main():
    parser = argparse.ArgumentParser(description="Evaluate VisionQC with labeled images.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--normal-dirs", nargs="+", required=True)
    parser.add_argument("--defect-dirs", nargs="+", required=True)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--thresholds", default=None, help="comma list, e.g. 30,40,50 (overrides config)")
    parser.add_argument("--min-area", type=int, default=None)
    parser.add_argument("--manual-as", choices=["defect", "normal"], default="defect")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    try:
        params = load_params(args.config)
        reference = load_image(args.reference)
    except ValueError as error:
        print(f"ERROR: {error}")
        sys.exit(1)
    if args.min_area is not None:
        params.min_area = args.min_area

    samples = [(p, 0) for d in args.normal_dirs for p in list_images(d)]
    samples += [(p, 1) for d in args.defect_dirs for p in list_images(d)]
    if not samples:
        print("ERROR: no images found. Check the folder paths.")
        sys.exit(1)

    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except OSError as error:
        print(f"Cannot create output folder: {error}")
        args.output_dir = None

    thresholds = [params.threshold] if not args.thresholds else [int(t) for t in args.thresholds.split(",")]
    total = 0
    for threshold in thresholds:
        params.threshold = threshold
        csv_path = os.path.join(args.output_dir, f"evaluation_t{threshold}.csv") if args.output_dir else None
        total = run_once(reference, samples, params.validated(), args.manual_as == "defect", csv_path)

    if total < 30:
        print("\nNOTE: fewer than 30 images were evaluated. These numbers describe this small set only "
              "and must not be presented as general performance.")


if __name__ == "__main__":
    main()