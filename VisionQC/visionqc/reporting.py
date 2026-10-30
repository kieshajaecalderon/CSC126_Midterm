"""Saving the annotated image plus JSON and CSV reports."""
import csv
import json
import os
from datetime import datetime

from .io_utils import save_image


def build_report(result, reference_path, test_path):
    """Create a plain-Python dictionary describing one inspection."""
    return {
        "inspection_time": datetime.now().isoformat(timespec="seconds"),
        "reference_file": os.path.basename(reference_path or ""),
        "test_file": os.path.basename(test_path or ""),
        "status": result.status,
        "reasons": list(result.reasons),
        "measurements": {
            "image_width_px": int(result.image_size[0]),
            "image_height_px": int(result.image_size[1]),
            "candidate_region_count": len(result.regions),
            "total_candidate_area_px": round(float(result.total_area), 2),
            "coverage_percent": round(float(result.coverage_percent), 4),
            "peak_difference_0_255": int(result.peak_difference),
            "brightness_shift_gray_levels": round(float(result.brightness_shift), 2),
            "alignment": result.alignment_message,
        },
        "parameters": result.params.to_dict(),
        "regions": [
            {
                "id": r.region_id, "x": int(r.x), "y": int(r.y),
                "width": int(r.w), "height": int(r.h),
                "area_px": round(float(r.area), 2),
                "mean_difference": round(float(r.mean_diff), 2),
            }
            for r in result.regions
        ],
        "notes": "All measurements are in image pixels, not physical units. "
                 "'Suspected Defect' is not a confirmed manufacturing defect.",
    }


def save_report_json(path, report):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)


def save_report_csv(path, report):
    """One row per region (or one summary row when there are no regions)."""
    header = ["inspection_time", "reference_file", "test_file", "status", "region_count",
              "total_area_px", "coverage_percent", "threshold", "min_area",
              "region_id", "x", "y", "width", "height", "area_px", "mean_difference"]
    m, p = report["measurements"], report["parameters"]
    base = [report["inspection_time"], report["reference_file"], report["test_file"],
            report["status"], m["candidate_region_count"], m["total_candidate_area_px"],
            m["coverage_percent"], p["threshold"], p["min_area"]]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        if not report["regions"]:
            writer.writerow(base + [""] * 7)
        for r in report["regions"]:
            writer.writerow(base + [r["id"], r["x"], r["y"], r["width"], r["height"],
                                    r["area_px"], r["mean_difference"]])


def save_inspection(output_dir, result, reference_path, test_path):
    """Save annotated image, mask, JSON and CSV. Returns the list of saved paths.
    Raises ValueError with a readable message if anything cannot be written."""
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as error:
        raise ValueError(f"Cannot create output folder '{output_dir}': {error}")

    stem = os.path.splitext(os.path.basename(test_path or "test"))[0]
    base = os.path.join(output_dir, f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    report = build_report(result, reference_path, test_path)
    saved = []
    try:
        save_image(base + "_annotated.png", result.annotated)
        saved.append(base + "_annotated.png")
        save_image(base + "_mask.png", result.final_mask)
        saved.append(base + "_mask.png")
        save_report_json(base + "_report.json", report)
        saved.append(base + "_report.json")
        save_report_csv(base + "_report.csv", report)
        saved.append(base + "_report.csv")
    except OSError as error:
        raise ValueError(f"Could not write results: {error}")
    return saved