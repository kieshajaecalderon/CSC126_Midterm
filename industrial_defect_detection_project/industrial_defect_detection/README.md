# Industrial Defect Detection System (CSC 126)

A student-friendly computer vision demo that uses **classical OpenCV techniques** to highlight regions that may be surface defects in industrial component images.

> Important: the program flags *suspected* defects. It is not a certified inspection tool, and textured surfaces, edges, reflections, and uneven lighting may create false positives.

## Computer vision topics demonstrated

1. **Image Operations:** grayscale conversion, CLAHE contrast enhancement, Gaussian blur, and absolute difference.
2. **Image Segmentation:** Otsu thresholding with an adjustable threshold bias.
3. **Image Analysis and Transformation:** morphological opening/closing, contour detection, area filtering, bounding boxes, and centroid measurements.

## Project structure

```text
industrial_defect_detection/
├── main.py
├── requirements.txt
├── README.md
├── samples/       # Add your own permitted sample images here
└── outputs/       # Annotated results and masks are saved here
```

## Requirements

- Python 3.10 or newer recommended
- OpenCV
- NumPy
- Tkinter for the optional file picker (usually included with standard desktop Python installations)

## Setup

Create and activate a virtual environment if desired.

**Windows**
```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

**macOS/Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

Open a file picker:
```bash
python main.py
```

Or specify an image directly:
```bash
python main.py --image samples/component.jpg
```

Tune detection parameters:
```bash
python main.py --image samples/component.jpg --min-area 30 --threshold-bias 5
```

- `--min-area`: ignore contours smaller than this area in pixels squared.
- `--max-area`: ignore contours larger than this area; `0` disables the maximum.
- `--threshold-bias`: negative values make the segmentation more sensitive; positive values make it less sensitive.
- `--blur-size`: odd Gaussian blur kernel size.
- `--morphology-size`: odd morphology kernel size.

## OpenCV window controls

- **O** — open another image
- **S** — save the annotated result and binary mask to `outputs/`
- **+ / =** — increase threshold bias (less sensitive)
- **- / _** — decrease threshold bias (more sensitive)
- **[ / ]** — decrease/increase minimum contour area
- **R** — reset adjustable settings
- **Q / Esc** — quit

When you press **S**, the program saves two files: an annotated image and a binary segmentation mask. It also prints the current region measurements in the terminal.

## How the algorithm works

1. Load the image and convert it to grayscale.
2. Apply CLAHE to improve local contrast.
3. Blur the enhanced grayscale image to estimate gradual background changes.
4. Calculate the absolute difference between the enhanced image and its blurred version.
5. Apply Otsu thresholding to create a binary mask of local intensity changes.
6. Use morphological opening and closing to remove isolated pixels and close small gaps.
7. Find external contours and filter them by area.
8. Draw contours, bounding boxes, and labels on the original image.

## Test plan

Test with:
- A normal component image.
- A component with a visible scratch, crack, hole, or discoloration.
- A low-contrast or unevenly lit image.
- A highly textured surface.
- An image expected to have no significant suspicious region.

Record what the program actually detects. Do not invent accuracy values. If you have ground-truth defect masks, evaluate detections against those labels; otherwise, discuss qualitative behavior and limitations.

## Limitations

- Local intensity changes are not necessarily defects.
- Thin scratches may be missed or fragmented.
- Surface texture can generate false positives.
- Lighting, image scale, camera angle, and object orientation affect results.
- The current implementation does not learn a normal-product model or classify defect types.
- Human review is required before treating any candidate as a real defect.

## Suggested improvements

- Restrict analysis to a manually selected region of interest.
- Compare an inspected part with a correctly aligned reference image.
- Add a small labelled test set and calculate precision/recall.
- Add a user interface for tuning thresholds and selecting regions.
- Test on a licensed industrial inspection dataset such as MVTec AD.

## GitHub submission checklist

- [ ] Include the complete source code.
- [ ] Include `requirements.txt` and this README.
- [ ] Add sample images only if their licenses permit redistribution.
- [ ] Add screenshots of actual program results.
- [ ] Do not commit virtual environments or huge datasets.
- [ ] Explain how to run the application and disclose its limitations.
