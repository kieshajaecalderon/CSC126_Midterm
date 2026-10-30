"""VisionQC entry point. Run with:  python main.py"""
import sys

try:
    import cv2  # noqa: F401
    import numpy  # noqa: F401
except ImportError:
    print("Missing packages. Run:  pip install -r requirements.txt")
    sys.exit(1)

import tkinter as tk
from visionqc.gui import VisionQCApp


def main():
    root = tk.Tk()
    VisionQCApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()