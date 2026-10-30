"""TOPIC 3 (part) - IMAGE TRANSFORMATION: align the test image to the reference.

Method: ORB keypoints -> brute-force matching -> RANSAC homography -> warpPerspective.
If alignment is not reliable, the test image is used unchanged and a message says so.
"""
import cv2
import numpy as np
from .preprocessing import to_grayscale


def align_to_reference(test_bgr, ref_bgr):
    """Return (aligned_test, valid_mask, message, success).

    valid_mask is 255 where the warped image has real pixels and 0 on empty borders,
    so empty borders are never counted as defects.
    """
    height, width = ref_bgr.shape[:2]
    full_mask = np.full((height, width), 255, np.uint8)

    ref_gray = to_grayscale(ref_bgr)
    test_gray = to_grayscale(test_bgr)

    orb = cv2.ORB_create(nfeatures=1500)
    keypoints_ref, descriptors_ref = orb.detectAndCompute(ref_gray, None)
    keypoints_test, descriptors_test = orb.detectAndCompute(test_gray, None)

    if descriptors_ref is None or descriptors_test is None:
        return test_bgr, full_mask, "Alignment skipped: too few image features found.", False

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(descriptors_test, descriptors_ref), key=lambda m: m.distance)
    matches = matches[:300]
    if len(matches) < 12:
        return test_bgr, full_mask, "Alignment skipped: too few feature matches.", False

    test_points = np.float32([keypoints_test[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    ref_points = np.float32([keypoints_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    homography, inlier_mask = cv2.findHomography(test_points, ref_points, cv2.RANSAC, 3.0)
    if homography is None or inlier_mask is None:
        return test_bgr, full_mask, "Alignment skipped: no valid transform found.", False

    inliers = int(inlier_mask.sum())
    if inliers < 12:
        return test_bgr, full_mask, f"Alignment skipped: only {inliers} reliable matches.", False

    # Sanity check: the transform must not shrink/stretch the image absurdly.
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners, homography)
    area_ratio = cv2.contourArea(warped_corners.reshape(-1, 2)) / float(width * height)
    if not 0.7 < area_ratio < 1.4:
        return test_bgr, full_mask, "Alignment skipped: transform looked implausible.", False

    # If images are already aligned, skip warping (avoids needless interpolation blur).
    max_shift = float(np.max(np.linalg.norm((warped_corners - corners).reshape(-1, 2), axis=1)))
    if max_shift < 1.5:
        return test_bgr, full_mask, f"Already aligned ({inliers} matches, no warp needed).", True

    aligned = cv2.warpPerspective(test_bgr, homography, (width, height))
    valid = cv2.warpPerspective(full_mask, homography, (width, height))
    valid = cv2.erode(valid, np.ones((5, 5), np.uint8))  # trim unreliable border pixels
    return aligned, valid, f"Aligned with homography ({inliers} matches).", True