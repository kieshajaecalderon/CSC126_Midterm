"""TOPIC 2 - IMAGE SEGMENTATION and TOPIC 3 (part) - IMAGE ANALYSIS.

tolerant difference -> threshold -> morphological open/close -> contours -> area filter
-> measurements
"""
from dataclasses import dataclass, field
import cv2
import numpy as np


@dataclass
class Region:
    region_id: int
    x: int
    y: int
    w: int
    h: int
    area: float            # contour area in PIXELS (not physical units)
    mean_diff: float       # average difference level inside the region (0-255)
    centroid: tuple
    contour: np.ndarray = field(repr=False, default=None)


def compute_difference(reference_pre, test_pre, tolerance=1):
    """Difference between reference and test (bright = big change).

    tolerance = 1: plain cv2.absdiff.
    tolerance > 1: shift-tolerant difference. The reference is expanded into a band
    [lower, upper] using erode/dilate (local minimum/maximum). A test pixel only counts
    as different if it falls OUTSIDE that band, so a tiny camera shift of about
    tolerance//2 pixels does not create false differences along edges and texture.
    """
    if tolerance <= 1:
        return cv2.absdiff(reference_pre, test_pre)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (tolerance, tolerance))
    upper = cv2.dilate(reference_pre, kernel)
    lower = cv2.erode(reference_pre, kernel)
    brighter_than_band = cv2.subtract(test_pre, upper)   # saturating subtraction
    darker_than_band = cv2.subtract(lower, test_pre)
    return cv2.max(brighter_than_band, darker_than_band)


def threshold_difference(diff, threshold):
    """cv2.threshold: pixels differing by more than `threshold` become 255 (white), else 0."""
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return mask


def clean_mask(mask, kernel_size):
    """Morphological opening removes tiny white specks (noise);
    closing fills small holes inside real regions."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


def extract_regions(mask, diff, min_area):
    """Find contours, keep those with area >= min_area, and measure them.

    Returns (regions sorted by area, filtered_mask containing only the kept regions).
    """
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    filtered_mask = np.zeros_like(mask)
    regions = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue  # too small: treated as noise, not a candidate defect

        x, y, w, h = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if moments["m00"] > 0:
            centroid = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
        else:
            centroid = (x + w // 2, y + h // 2)

        single = np.zeros_like(mask)
        cv2.drawContours(single, [contour], -1, 255, thickness=cv2.FILLED)
        mean_diff = float(cv2.mean(diff, mask=single)[0])

        cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)
        regions.append(Region(0, x, y, w, h, float(area), mean_diff, centroid, contour))

    regions.sort(key=lambda r: r.area, reverse=True)
    for number, region in enumerate(regions, start=1):
        region.region_id = number
    return regions, filtered_mask