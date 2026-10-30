"""Main inspection pipeline and the decision logic."""
from dataclasses import dataclass, field
import cv2
import numpy as np

from .alignment import align_to_reference
from .preprocessing import resize_to_width, preprocess, to_grayscale
from .segmentation import compute_difference, threshold_difference, clean_mask, extract_regions

STATUS_NORMAL = "Likely Normal"
STATUS_DEFECT = "Suspected Defect"
STATUS_REVIEW = "Manual Review Required"


@dataclass
class InspectionResult:
    status: str
    reasons: list
    regions: list
    total_area: float
    coverage_percent: float
    image_size: tuple
    alignment_message: str
    brightness_shift: float
    peak_difference: int
    params: object
    reference_view: np.ndarray = field(repr=False, default=None)
    test_view: np.ndarray = field(repr=False, default=None)
    test_preprocessed: np.ndarray = field(repr=False, default=None)
    diff_display: np.ndarray = field(repr=False, default=None)
    cleaned_mask: np.ndarray = field(repr=False, default=None)
    final_mask: np.ndarray = field(repr=False, default=None)
    annotated: np.ndarray = field(repr=False, default=None)


def _check_image(image, name):
    if image is None or not hasattr(image, "shape") or image.size == 0:
        raise ValueError(f"The {name} image is empty or missing.")


def _match_brightness(reference_pre, test_pre, valid_mask):
    """Shift the test image so its MEDIAN brightness equals the reference's.
    The median is used because a defect barely changes it (a mean would be pulled by it)."""
    valid = valid_mask > 0
    if not valid.any():
        return test_pre
    offset = float(np.median(reference_pre[valid])) - float(np.median(test_pre[valid]))
    shifted = test_pre.astype(np.int16) + int(round(offset))
    return np.clip(shifted, 0, 255).astype(np.uint8)


def _annotate(test_view, final_mask, regions):
    """Red translucent overlay + red outline + yellow bounding box + label."""
    overlay = test_view.copy()
    overlay[final_mask > 0] = (0, 0, 255)
    annotated = cv2.addWeighted(overlay, 0.35, test_view, 0.65, 0)
    for region in regions:
        cv2.drawContours(annotated, [region.contour], -1, (0, 0, 255), 2)
        cv2.rectangle(annotated, (region.x, region.y),
                      (region.x + region.w, region.y + region.h), (0, 255, 255), 1)
        cv2.putText(annotated, f"#{region.region_id}", (region.x, max(12, region.y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return annotated


def inspect_pair(reference_bgr, test_bgr, params):
    """Compare a test image with the reference image and return an InspectionResult."""
    params = params.validated()
    _check_image(reference_bgr, "reference")
    _check_image(test_bgr, "test")

    unreliable = []  # reasons why the result cannot be trusted

    # --- Handle different image dimensions -------------------------------------------
    ref_h, ref_w = reference_bgr.shape[:2]
    test_h, test_w = test_bgr.shape[:2]
    if abs((ref_w / ref_h) / (test_w / test_h) - 1.0) > 0.05:
        unreliable.append("Reference and test images have different aspect ratios "
                          "(possibly different product or camera setup).")

    reference_view = resize_to_width(reference_bgr, params.work_width)
    height, width = reference_view.shape[:2]
    interpolation = cv2.INTER_AREA if test_w > width else cv2.INTER_LINEAR
    test_view = cv2.resize(test_bgr, (width, height), interpolation=interpolation)

    # --- TOPIC 3: alignment (optional) -----------------------------------------------
    if params.use_alignment:
        test_view, valid_mask, alignment_message, _ = align_to_reference(test_view, reference_view)
    else:
        valid_mask = np.full((height, width), 255, np.uint8)
        alignment_message = "Alignment disabled (fixed-view comparison)."

    # --- TOPIC 1: image operations ---------------------------------------------------
    reference_pre = preprocess(reference_view, params)
    test_pre = preprocess(test_view, params)

    ref_brightness = cv2.mean(to_grayscale(reference_view), mask=valid_mask)[0]
    test_brightness = cv2.mean(to_grayscale(test_view), mask=valid_mask)[0]
    brightness_shift = abs(ref_brightness - test_brightness)

    if params.normalize_brightness:
        test_pre = _match_brightness(reference_pre, test_pre, valid_mask)

    # --- TOPIC 2: segmentation -------------------------------------------------------
    diff = compute_difference(reference_pre, test_pre, params.shift_tolerance)
    diff = cv2.bitwise_and(diff, diff, mask=valid_mask)       # ignore empty warped borders
    peak_difference = int(diff.max())
    raw_mask = threshold_difference(diff, params.threshold)
    cleaned_mask = clean_mask(raw_mask, params.morph_kernel)

    # --- TOPIC 3: contour analysis and measurements ----------------------------------
    regions, final_mask = extract_regions(cleaned_mask, diff, params.min_area)
    image_pixels = float(width * height)
    total_area = float(sum(region.area for region in regions))
    coverage_percent = 100.0 * total_area / image_pixels
    cleaned_coverage = 100.0 * np.count_nonzero(cleaned_mask) / image_pixels

    # --- Decision ----------------------------------------------------------------------
    if brightness_shift > params.max_brightness_shift:
        unreliable.append(f"Lighting differs strongly (mean brightness shift "
                          f"{brightness_shift:.1f} > {params.max_brightness_shift:.1f}).")
    if cleaned_coverage > params.max_coverage_percent:
        unreliable.append(f"Too much of the image differs ({cleaned_coverage:.1f}% > "
                          f"{params.max_coverage_percent:.1f}%): likely misalignment, wrong product, "
                          f"or lighting change.")
    if len(regions) > params.max_regions:
        unreliable.append(f"Too many candidate regions ({len(regions)} > {params.max_regions}): "
                          f"likely noise or misalignment.")

    if unreliable:
        status, reasons = STATUS_REVIEW, unreliable
    elif regions:
        status = STATUS_DEFECT
        reasons = [f"{len(regions)} region(s) have area >= {params.min_area} px after "
                   f"thresholding at {params.threshold}. This is a SUSPICION for a human "
                   f"inspector, not a confirmed defect."]
    else:
        status = STATUS_NORMAL
        reasons = [f"No region reached {params.min_area} px at threshold {params.threshold}. "
                   f"Strongest difference found: {peak_difference} (0-255). "
                   f"If you can see a defect, lower the threshold or minimum area. "
                   f"If this value is near 0, check that the test and reference are different files."]

    annotated = _annotate(test_view, final_mask, regions)
    diff_display = cv2.convertScaleAbs(diff, alpha=3.0)  # brightened for viewing only

    return InspectionResult(
        status=status, reasons=reasons, regions=regions, total_area=total_area,
        coverage_percent=coverage_percent, image_size=(width, height),
        alignment_message=alignment_message, brightness_shift=float(brightness_shift),
        peak_difference=peak_difference, params=params, reference_view=reference_view,
        test_view=test_view, test_preprocessed=test_pre, diff_display=diff_display,
        cleaned_mask=cleaned_mask, final_mask=final_mask, annotated=annotated,
    )