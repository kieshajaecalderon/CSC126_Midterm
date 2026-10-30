"""Inspection parameters and loading/saving of config.json."""
import json
import os
from dataclasses import dataclass, asdict, fields


@dataclass
class InspectionParams:
    work_width: int = 512               # both images are resized to this width
    blur_kernel: int = 5                # Gaussian blur size (odd number)
    use_clahe: bool = True              # contrast enhancement on/off
    clahe_clip_limit: float = 2.0
    use_alignment: bool = True          # ORB + homography alignment on/off
    normalize_brightness: bool = True   # match test brightness to the reference
    shift_tolerance: int = 3            # odd; ignores shifts of about tolerance//2 pixels (1 = off)
    threshold: int = 25                 # difference level (0-255) counted as "different"
    morph_kernel: int = 3               # size of the morphological kernel
    min_area: int = 100                 # smallest region (pixels) kept as a candidate
    max_coverage_percent: float = 25.0  # above this, result is unreliable
    max_regions: int = 20               # more regions than this, result is unreliable
    max_brightness_shift: float = 40.0  # lighting difference limit (gray levels)

    def validated(self):
        """Return a copy with every value forced into a safe range."""
        def odd(value):
            value = max(1, int(value))
            return value if value % 2 == 1 else value + 1

        return InspectionParams(
            work_width=max(64, int(self.work_width)),
            blur_kernel=odd(self.blur_kernel),
            use_clahe=bool(self.use_clahe),
            clahe_clip_limit=max(0.5, float(self.clahe_clip_limit)),
            use_alignment=bool(self.use_alignment),
            normalize_brightness=bool(self.normalize_brightness),
            shift_tolerance=min(15, odd(self.shift_tolerance)),
            threshold=min(255, max(1, int(self.threshold))),
            morph_kernel=max(1, int(self.morph_kernel)),
            min_area=max(1, int(self.min_area)),
            max_coverage_percent=min(100.0, max(0.1, float(self.max_coverage_percent))),
            max_regions=max(1, int(self.max_regions)),
            max_brightness_shift=max(1.0, float(self.max_brightness_shift)),
        )

    def to_dict(self):
        return asdict(self)


def load_params(path):
    """Load parameters from a JSON file. Missing file -> defaults.
    Raises ValueError when the file exists but cannot be understood."""
    if not os.path.isfile(path):
        return InspectionParams()
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read config file '{path}': {error}")
    valid_names = {f.name for f in fields(InspectionParams)}
    known = {key: value for key, value in data.items() if key in valid_names}
    try:
        return InspectionParams(**known).validated()
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid value in config file '{path}': {error}")