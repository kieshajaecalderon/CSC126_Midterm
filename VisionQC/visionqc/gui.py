"""Compact Tkinter interface for VisionQC (sized for a 13-inch laptop screen).
Tip: double-click any small view to open it larger."""
import base64
import os
import traceback
import tkinter as tk
from dataclasses import replace
from tkinter import ttk, filedialog, messagebox

import cv2

from .config import InspectionParams, load_params
from .inspector import inspect_pair, STATUS_NORMAL, STATUS_DEFECT
from .io_utils import load_image
from .reporting import save_inspection

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FILE_TYPES = [("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")]
ENLARGED_BOX = (760, 520)
STATUS_COLORS = {STATUS_NORMAL: "#1b7f3b", STATUS_DEFECT: "#c62828"}
REVIEW_COLOR = "#e08600"

VIEW_TITLES = {
    "reference": "Reference",
    "test": "Test image",
    "preprocessed": "Preprocessed [Image Ops]",
    "diff": "Difference x3 [Analysis]",
    "mask": "Mask [Segmentation]",
    "annotated": "Result",
}


class VisionQCApp:
    def __init__(self, root):
        self.root = root
        root.title("VisionQC - Visual Quality Control")

        # Size the views from the screen width so everything fits on a small laptop.
        screen_width = root.winfo_screenwidth()
        self.view_w = max(150, min(230, (screen_width - 280) // 3 - 12))
        self.view_h = self.view_w * 3 // 4

        try:
            self.base_params = load_params(CONFIG_PATH)
        except ValueError as error:
            messagebox.showwarning("Config problem", f"{error}\nDefault values will be used.")
            self.base_params = InspectionParams()

        self.reference_path = self.test_path = None
        self.reference_image = self.test_image = None
        self.result = None
        self._photos = {}        # keeps PhotoImage objects alive
        self._view_images = {}   # raw images, used for the enlarged window
        self.view_labels = {}

        self._build_variables()
        self._build_layout()
        self._reset_views()

    # ------------------------------------------------------------------ layout
    def _build_variables(self):
        p = self.base_params
        self.var_threshold = tk.IntVar(value=p.threshold)
        self.var_min_area = tk.IntVar(value=p.min_area)
        self.var_tolerance = tk.IntVar(value=p.shift_tolerance)
        self.var_clahe = tk.BooleanVar(value=p.use_clahe)
        self.var_align = tk.BooleanVar(value=p.use_alignment)

    def _build_layout(self):
        left = ttk.Frame(self.root, padding=6)
        left.grid(row=0, column=0, sticky="ns")
        right = ttk.Frame(self.root, padding=6)
        right.grid(row=0, column=1, sticky="nsew")
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        files = ttk.LabelFrame(left, text="Files", padding=4)
        files.pack(fill="x", pady=(0, 4))
        ttk.Button(files, text="1. Load Reference", command=self.load_reference).pack(fill="x")
        self.reference_label = ttk.Label(files, text="(none)", wraplength=190)
        self.reference_label.pack(anchor="w")
        ttk.Button(files, text="2. Load Test Image", command=self.load_test).pack(fill="x", pady=(4, 0))
        self.test_label = ttk.Label(files, text="(none)", wraplength=190)
        self.test_label.pack(anchor="w")

        params = ttk.LabelFrame(left, text="Parameters (release slider)", padding=4)
        params.pack(fill="x", pady=(0, 4))
        self._add_scale(params, "Threshold (lower = more sensitive)", self.var_threshold, 1, 150)
        self._add_scale(params, "Minimum region area (px)", self.var_min_area, 1, 2000)
        self._add_scale(params, "Shift tolerance (1 = off)", self.var_tolerance, 1, 9, resolution=2)
        ttk.Checkbutton(params, text="Contrast enhancement (CLAHE)", variable=self.var_clahe,
                        command=self._on_param_change).pack(anchor="w")
        ttk.Checkbutton(params, text="Align test to reference", variable=self.var_align,
                        command=self._on_param_change).pack(anchor="w")
        ttk.Button(params, text="Reset to config.json", command=self.reset_params).pack(fill="x", pady=(4, 0))

        actions = ttk.LabelFrame(left, text="Actions", padding=4)
        actions.pack(fill="x", pady=(0, 4))
        ttk.Button(actions, text="3. Run Inspection", command=self.run_inspection).pack(fill="x")
        ttk.Button(actions, text="4. Save Results", command=self.save_results).pack(fill="x", pady=(3, 0))

        ttk.Label(left, text="Same product, position, lighting and camera for both images. "
                             "Double-click a view to enlarge it.",
                  wraplength=200, foreground="#555").pack(anchor="w")

        grid = ttk.Frame(right)
        grid.pack(anchor="w")
        for index, key in enumerate(VIEW_TITLES):
            frame = ttk.LabelFrame(grid, text=VIEW_TITLES[key], padding=1)
            frame.grid(row=index // 3, column=index % 3, padx=2, pady=2)
            box = tk.Frame(frame, width=self.view_w, height=self.view_h, bg="#222222")
            box.pack_propagate(False)
            box.pack()
            label = tk.Label(box, bg="#222222")
            label.pack(expand=True)
            label.bind("<Double-Button-1>", lambda _e, k=key: self._enlarge(k))
            self.view_labels[key] = label

        self.status_label = tk.Label(right, text="Status: waiting for images",
                                     font=("Segoe UI", 12, "bold"), anchor="w")
        self.status_label.pack(fill="x", pady=(4, 2))
        text_frame = ttk.Frame(right)
        text_frame.pack(fill="both", expand=True)
        self.results_text = tk.Text(text_frame, height=8, width=70, wrap="word",
                                    state="disabled", font=("Consolas", 9))
        scrollbar = ttk.Scrollbar(text_frame, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        self.results_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _add_scale(self, parent, text, variable, low, high, resolution=1):
        ttk.Label(parent, text=text).pack(anchor="w")
        scale = tk.Scale(parent, from_=low, to=high, orient="horizontal", variable=variable,
                         resolution=resolution, length=190, width=12)
        scale.pack(fill="x")
        scale.bind("<ButtonRelease-1>", self._on_param_change)
        scale.bind("<KeyRelease>", self._on_param_change)

    # ---------------------------------------------------------------- helpers
    def _read_params(self):
        return replace(
            self.base_params,
            threshold=self.var_threshold.get(), min_area=self.var_min_area.get(),
            shift_tolerance=self.var_tolerance.get(),
            use_clahe=self.var_clahe.get(), use_alignment=self.var_align.get(),
        ).validated()

    @staticmethod
    def _to_photo(image, box):
        height, width = image.shape[:2]
        scale = min(box[0] / width, box[1] / height)
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        small = cv2.resize(image, size, interpolation=interpolation)
        _, buffer = cv2.imencode(".png", small)
        return tk.PhotoImage(data=base64.b64encode(buffer.tobytes()).decode("ascii"))

    def _show(self, key, image):
        if image is None:
            self.view_labels[key].configure(image="")
            self._photos.pop(key, None)
            self._view_images.pop(key, None)
            return
        photo = self._to_photo(image, (self.view_w, self.view_h))
        self._photos[key] = photo
        self._view_images[key] = image
        self.view_labels[key].configure(image=photo)

    def _enlarge(self, key):
        image = self._view_images.get(key)
        if image is None:
            return
        window = tk.Toplevel(self.root)
        window.title(VIEW_TITLES[key])
        photo = self._to_photo(image, ENLARGED_BOX)
        label = tk.Label(window, image=photo)
        label.image = photo   # keep a reference so the image is not garbage-collected
        label.pack()

    def _reset_views(self):
        for key in VIEW_TITLES:
            self._show(key, None)
        if self.reference_image is not None:
            self._show("reference", self.reference_image)
        if self.test_image is not None:
            self._show("test", self.test_image)

    def _set_results(self, text):
        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", text)
        self.results_text.configure(state="disabled")

    # ---------------------------------------------------------------- actions
    def load_reference(self):
        path = filedialog.askopenfilename(title="Select the NORMAL reference image", filetypes=FILE_TYPES)
        if not path:
            return
        try:
            image = load_image(path)
        except ValueError as error:
            messagebox.showerror("Cannot load reference image", str(error))
            return
        self.reference_path, self.reference_image = path, image
        self.reference_label.configure(text=os.path.basename(path))
        self._show("reference", image)
        if self.test_image is not None:
            self.run_inspection()

    def load_test(self):
        path = filedialog.askopenfilename(title="Select the TEST image", filetypes=FILE_TYPES)
        if not path:
            return
        try:
            image = load_image(path)
        except ValueError as error:
            messagebox.showerror("Cannot load test image", str(error))
            return
        self.test_path, self.test_image = path, image
        self.test_label.configure(text=os.path.basename(path))
        self.result = None
        self._reset_views()
        if self.reference_image is not None:
            self.run_inspection()   # same reference + same parameters, no restart needed

    def reset_params(self):
        p = self.base_params
        self.var_threshold.set(p.threshold)
        self.var_min_area.set(p.min_area)
        self.var_tolerance.set(p.shift_tolerance)
        self.var_clahe.set(p.use_clahe)
        self.var_align.set(p.use_alignment)
        self._on_param_change()

    def _on_param_change(self, _event=None):
        if self.reference_image is not None and self.test_image is not None:
            self.run_inspection()

    def run_inspection(self):
        if self.reference_image is None:
            messagebox.showwarning("Missing reference", "Please load a reference image first.")
            return
        if self.test_image is None:
            messagebox.showwarning("Missing test image", "Please load a test image first.")
            return
        try:
            result = inspect_pair(self.reference_image, self.test_image, self._read_params())
        except ValueError as error:
            messagebox.showerror("Inspection failed", str(error))
            return
        except Exception as error:  # unexpected problem: report, do not crash
            traceback.print_exc()
            messagebox.showerror("Unexpected error", f"{type(error).__name__}: {error}")
            return

        self.result = result
        self._show("reference", result.reference_view)
        self._show("test", result.test_view)
        self._show("preprocessed", result.test_preprocessed)
        self._show("diff", result.diff_display)
        self._show("mask", result.final_mask)
        self._show("annotated", result.annotated)
        self._update_results(result)

    def _update_results(self, r):
        color = STATUS_COLORS.get(r.status, REVIEW_COLOR)
        self.status_label.configure(
            text=f"Status: {r.status}  |  Defect regions: {len(r.regions)}", fg=color)
        p = r.params
        lines = [
            *[f"- {reason}" for reason in r.reasons],
            "",
            "Measurements (pixels, not calibrated to physical size):",
            f"  Image size {r.image_size[0]}x{r.image_size[1]} | regions {len(r.regions)} | "
            f"area {r.total_area:.0f} px | coverage {r.coverage_percent:.3f}%",
            f"  Strongest difference {r.peak_difference}/255 | brightness shift "
            f"{r.brightness_shift:.1f} | {r.alignment_message}",
        ]
        for region in r.regions:
            lines.append(f"  #{region.region_id}: x={region.x} y={region.y} w={region.w} h={region.h} "
                         f"area={region.area:.0f}px meandiff={region.mean_diff:.1f}")
        lines += [
            "",
            "Rule: Manual Review if limits are exceeded; else Suspected Defect if a region >= "
            "min area; else Likely Normal.",
            f"Parameters: threshold={p.threshold}, min_area={p.min_area}, "
            f"shift_tol={p.shift_tolerance}, CLAHE={p.use_clahe}, align={p.use_alignment}",
        ]
        self._set_results("\n".join(lines))

    def save_results(self):
        if self.result is None:
            messagebox.showwarning("Nothing to save", "Run an inspection first.")
            return
        try:
            os.makedirs(RESULTS_DIR, exist_ok=True)
        except OSError:
            pass  # the folder chooser below still lets the user pick another place
        folder = filedialog.askdirectory(
            title="Choose a folder for the results",
            initialdir=RESULTS_DIR if os.path.isdir(RESULTS_DIR) else BASE_DIR)
        if not folder:
            return
        try:
            saved = save_inspection(folder, self.result, self.reference_path, self.test_path)
        except ValueError as error:
            messagebox.showerror("Save failed", str(error))
            return
        messagebox.showinfo("Saved", "Saved files:\n" + "\n".join(os.path.basename(p) for p in saved))