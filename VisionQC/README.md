# VisionQC - Visual Quality Control (CSC 126 Midterm)

VisionQC compares a **test image** with a **normal reference image that you provide**,
highlights suspicious regions, counts and measures them, and reports one of three statuses.
It uses classical computer vision only (OpenCV + NumPy). No deep learning, no cloud, no GPU,
and no machine learning at all: every step is a fixed image-processing operation with
adjustable parameters. The program never generates a reference image.

## Features
- Load reference and test images (PNG, JPG, BMP, TIFF) with clear error messages
- Six compact views: reference, test, preprocessed test, difference, mask, annotated result
  (double-click a view to enlarge it)
- Adjustable threshold, minimum area, shift tolerance, CLAHE and alignment toggles
- Inspect many test images in one session with the same reference and parameters
- Save annotated image, mask, JSON report and CSV report
- Synthetic sample generator and an evaluation script (accuracy, precision, recall, F1)
- The window is sized to fit a 13-inch laptop screen

## Installation (Windows, PowerShell)
1. Install Python 3.9+ from python.org (tick "Add Python to PATH", keep "tcl/tk" selected).
2. In the project folder:
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
3. Run: `python main.py`

Always type each command on ONE line.

## Usage
1. `python generate_samples.py` creates synthetic images in `data/samples/`.
2. `python main.py`
3. **Load Reference** -> `data/samples/reference/reference.png`
4. **Load Test Image** -> any file in `data/samples/test/...`. Inspection runs automatically.
5. Move a slider and release it to re-inspect with new parameters.
6. **Save Results** writes the image and reports (default folder `results/`).
7. Load another test image to inspect the next product without restarting.

The reference and test must show the same product type, with similar position, orientation,
lighting and camera. For real photos, fix the camera (tripod or a stack of books) and take
the clean reference first.

## Where the three CSC 126 topics are implemented
| Topic | Functions (file) | Shown in view |
|---|---|---|
| Image Operations | `cv2.resize`, `cv2.cvtColor`, `cv2.GaussianBlur`, `cv2.createCLAHE` (`preprocessing.py`); median brightness matching (`inspector.py`); `cv2.morphologyEx` (`segmentation.py`) | Preprocessed |
| Image Segmentation | `cv2.dilate`/`cv2.erode` tolerant difference, `cv2.subtract`, `cv2.threshold`, `cv2.getStructuringElement`, `cv2.morphologyEx` open/close, `cv2.drawContours` filled (`segmentation.py`) | Difference, Mask |
| Image Analysis & Transformation | `cv2.ORB_create`, `cv2.BFMatcher`, `cv2.findHomography`, `cv2.warpPerspective` (`alignment.py`); `cv2.findContours`, `cv2.contourArea`, `cv2.boundingRect`, `cv2.moments`, `cv2.mean` (`segmentation.py`) | Test, Result |

Why each step: grayscale gives one brightness value per pixel; blur suppresses sensor noise;
CLAHE makes faint defects more visible; brightness matching removes overall lighting
differences; alignment removes small shifts between photos; the shift-tolerant difference
ignores sub-pixel to 1-2 pixel leftovers; the threshold decides which changes count; opening
removes specks and closing fills gaps; the area filter ignores regions that are too small.

## How the decision is made (`inspector.py`)
1. **Manual Review Required** if any of these is true: aspect ratios differ by more than 5%;
   mean brightness differs by more than `max_brightness_shift`; the cleaned mask covers more
   than `max_coverage_percent` of the image; more than `max_regions` candidates were found.
2. Otherwise **Suspected Defect** if at least one region has area >= `min_area`.
3. Otherwise **Likely Normal**. The result text shows the strongest difference found, so you
   can tell whether the threshold is too high.

"Suspected Defect" is a flag for a human inspector. It is not a confirmed defect.

## If a visible defect is reported as Likely Normal
- Check that the test and reference are two different files.
- Look at the "Strongest difference" value. If it is below the threshold, lower the threshold.
- Lower the minimum area for small defects.
- Thin or very faint defects may be missed. This is a limitation of simple differencing.
- Make sure the reference itself has no defect.

## Testing procedure
1. `python generate_samples.py`
2. Open the GUI, inspect one normal and one defective sample, and check the views.
3. Evaluate (one line each):
```
python evaluate.py --reference data/samples/reference/reference.png --normal-dirs data/samples/test/normal --defect-dirs data/samples/test/defective
python evaluate.py --reference data/samples/reference/reference.png --normal-dirs data/samples/test/normal --defect-dirs data/samples/test/defective --thresholds 10,15,20,25,30,40,60
```
Record the printed numbers yourself. This README contains no results because none were
measured while writing it.

**Metrics** (defective = positive): accuracy = (TP+TN)/all; precision = TP/(TP+FP);
recall = TP/(TP+FN); F1 = 2PR/(P+R).
- **False positive**: a normal product flagged as suspicious (wasted inspection time).
- **False negative**: a defective product passed as normal (more serious in manufacturing).

With only 28 synthetic images, results show that the pipeline works on controlled data.
They say nothing about real-world performance.

## Optional: MVTec AD
Download from the official site: https://www.mvtec.com/company/research/datasets/mvtec-ad
(accept their license, which restricts commercial use; do not redistribute the images).
Extract one category, e.g. to `data/mvtec_ad/transistor/`. Use one `train/good` image as the
reference, `test/good` as normal, and the other `test/<defect>` folders as defective (see
the `evaluate.py` docstring). A single reference suits objects that sit in a consistent
position; it performs poorly on random-texture categories (carpet, wood) and on objects that
rotate between shots.

## Limitations
- Needs a consistent view of the product; alignment is feature-based and can be skipped when
  few features exist (the result text says so).
- Lighting changes, shadows and reflections can cause false positives.
- Natural variation between two valid products looks like a defect.
- Faint or thin defects can be missed; sensitive settings can create false positives.
- All measurements are in pixels. There is no physical calibration.
- It cannot classify the defect type or confirm that a defect is real.
- Default parameters are starting points and must be tuned per product.

## 15-minute demo outline
1. Problem and idea (2 min) -> 2. Pipeline and the three topics (3 min) ->
3. Live demo: normal sample, defective sample, slider changes (5 min) ->
4. Save results, show the JSON/CSV (1 min) -> 5. Run `evaluate.py`, discuss FP/FN and
limitations honestly (4 min).