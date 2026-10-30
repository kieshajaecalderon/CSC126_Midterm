"""Create a small SYNTHETIC test set (no external dataset needed).

Output:
  data/samples/reference/reference.png
  data/samples/test/normal/normal_XX.png          (label: normal)
  data/samples/test/defective/defect_TYPE_XX.png  (label: defective)

Normal images = same product + sensor noise + small brightness change.
Defective images = normal image + one synthetic defect. Four defect types are obvious
(scratch, stain, hole, dent) and two are deliberately FAINT (faint_stain, thin_scratch),
so the evaluation can show misses instead of a trivial perfect score.
Run:  python generate_samples.py
"""
import os
import cv2
import numpy as np
from visionqc.io_utils import save_image

WIDTH, HEIGHT = 512, 384
OUT = os.path.join("data", "samples")


def make_base_product():
    """A gray metal-like plate with fixed texture, four screw holes and a ring."""
    rng = np.random.default_rng(0)
    image = np.full((HEIGHT, WIDTH), 90, np.float32)
    cv2.rectangle(image, (40, 30), (WIDTH - 40, HEIGHT - 30), 150, -1)
    texture = cv2.GaussianBlur(rng.normal(0, 6, (HEIGHT, WIDTH)).astype(np.float32), (0, 0), 1.5) * 2.5
    image += texture
    for cx, cy in [(70, 60), (WIDTH - 70, 60), (70, HEIGHT - 60), (WIDTH - 70, HEIGHT - 60)]:
        cv2.circle(image, (cx, cy), 10, 70, -1)
    cv2.circle(image, (WIDTH // 2, HEIGHT // 2), 50, 110, 3)
    cv2.rectangle(image, (200, 70), (312, 95), 120, -1)
    gray = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def make_normal(base, rng):
    """Same product, new 'photo': brightness shift and noise."""
    shifted = base.astype(np.float32) + rng.uniform(-4, 4)
    noisy = shifted + rng.normal(0, 3, base.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_scratch(img, rng):
    x, y = int(rng.integers(100, 300)), int(rng.integers(90, 290))
    dx, dy = int(rng.integers(60, 110)), int(rng.integers(-40, 40))
    cv2.line(img, (x, y), (x + dx, y + dy), (235, 235, 235), 4, cv2.LINE_AA)


def add_stain(img, rng):
    center = (int(rng.integers(120, 390)), int(rng.integers(90, 290)))
    axes = (int(rng.integers(16, 28)), int(rng.integers(10, 18)))
    layer = img.copy()
    cv2.ellipse(layer, center, axes, int(rng.integers(0, 180)), 0, 360, (40, 60, 90), -1)
    layer = cv2.GaussianBlur(layer, (5, 5), 0)
    img[:] = cv2.addWeighted(layer, 0.85, img, 0.15, 0)


def add_hole(img, rng):
    center = (int(rng.integers(120, 390)), int(rng.integers(90, 290)))
    cv2.circle(img, center, int(rng.integers(10, 15)), (15, 15, 15), -1, cv2.LINE_AA)


def add_dent(img, rng):
    center = (int(rng.integers(120, 390)), int(rng.integers(90, 290)))
    mask = np.zeros(img.shape[:2], np.float32)
    cv2.circle(mask, center, 20, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 6)[..., None]
    img[:] = np.clip(img.astype(np.float32) * (1 - 0.5 * mask), 0, 255).astype(np.uint8)


def add_faint_stain(img, rng):
    """Low-contrast darker patch (about 35 gray levels at the center)."""
    center = (int(rng.integers(120, 390)), int(rng.integers(90, 290)))
    axes = (int(rng.integers(22, 34)), int(rng.integers(14, 22)))
    layer = np.zeros(img.shape[:2], np.float32)
    cv2.ellipse(layer, center, axes, int(rng.integers(0, 180)), 0, 360, 1.0, -1)
    layer = cv2.GaussianBlur(layer, (0, 0), 4)[..., None]
    img[:] = np.clip(img.astype(np.float32) - 35 * layer, 0, 255).astype(np.uint8)


def add_thin_scratch(img, rng):
    """Thin (3 px), low-contrast scratch. The hardest sample type."""
    x, y = int(rng.integers(100, 300)), int(rng.integers(90, 290))
    dx, dy = int(rng.integers(60, 110)), int(rng.integers(-40, 40))
    cv2.line(img, (x, y), (x + dx, y + dy), (185, 185, 185), 3, cv2.LINE_AA)


DEFECTS = {"scratch": add_scratch, "stain": add_stain, "hole": add_hole, "dent": add_dent,
           "faint_stain": add_faint_stain, "thin_scratch": add_thin_scratch}


def main():
    rng = np.random.default_rng(42)
    base = make_base_product()
    for folder in ("reference", os.path.join("test", "normal"), os.path.join("test", "defective")):
        os.makedirs(os.path.join(OUT, folder), exist_ok=True)

    save_image(os.path.join(OUT, "reference", "reference.png"), base)

    for i in range(1, 11):
        save_image(os.path.join(OUT, "test", "normal", f"normal_{i:02d}.png"), make_normal(base, rng))

    count = 0
    for name, function in DEFECTS.items():
        for i in range(1, 4):
            image = make_normal(base, rng)
            function(image, rng)
            count += 1
            save_image(os.path.join(OUT, "test", "defective", f"defect_{name}_{i:02d}.png"), image)

    print(f"Created reference + 10 normal + {count} defective images in '{OUT}'.")


if __name__ == "__main__":
    main()