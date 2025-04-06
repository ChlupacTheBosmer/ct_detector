# ct_detector/data/utils.py

import os
import numpy as np
from pathlib import Path
from typing import List, Union, Optional
import traceback

from ultralytics.data.utils import img2label_paths
from ultralytics.engine.results import Boxes
from ultralytics.utils.ops import xywhn2xyxy

from ct_detector.utils.files import load_image


def get_image_paths_recursive(folder: Union[str, Path],
                               valid_exts: set = {'.jpg', '.jpeg', '.png'}) -> List[Path]:
    """
    Recursively find all image files with valid extensions in a folder.
    """
    folder = Path(folder)
    return [p for p in folder.rglob("*") if p.suffix.lower() in valid_exts and p.is_file()]


def filter_paths_by_name(paths: List[Union[str, Path]],
                          exclude_names_file: Union[str, Path]) -> List[Path]:
    """
    Filter out image paths whose filenames are listed in a given .txt file.
    """
    exclude_names_file = Path(exclude_names_file)
    with open(exclude_names_file, 'r') as f:
        exclude_names = set(Path(line.strip()).name for line in f if line.strip())
    return [Path(p) for p in paths if Path(p).name not in exclude_names]


def load_labels_for_images(img_path: Union[str, Path]) -> Optional[dict]:
    """
    Construct metadata dictionary for a given image including label path and Boxes.
    """
    img_path = Path(img_path)
    if not img_path.exists():
        return None

    # Primary: path from img2label_paths
    label_path = Path(img2label_paths([str(img_path)])[0])

    # Fallback: same directory with .txt extension
    if not label_path.exists():
        label_path = img_path.with_suffix(".txt")

    print(f"Loading labels for {img_path} from {label_path}")

    boxes = None
    labels = []
    if label_path.exists():
        try:
            img = load_image(str(img_path))[0]
            h, w = img.shape[:2]

            # Load raw YOLO-format boxes
            with open(label_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            raw = [[float(v) for v in line.split()] for line in lines]
            labels = [int(row[0]) for row in raw]
            data = [[*row[1:], 0.0, row[0]] for row in raw]  # class_id will be filled by 'labels', so set to dummy

            # Construct Boxes object
            if data:

                # Process box-by-box
                converted = []
                for row in np.array(data):
                    xywhn = row[:4]
                    cls_conf = row[4:]
                    xyxy = xywhn2xyxy(xywhn, w=640, h=640)
                    converted.append(np.concatenate([xyxy, cls_conf]))

                # Final array: shape (N, 6) with [x1, y1, x2, y2, cls, conf]
                converted_data = np.array(converted)

                boxes = Boxes(boxes=converted_data, orig_shape=(h, w))
            else:
                boxes = None
        except Exception as e:
            print(f"Error {e}")
            traceback.print_exc()
            boxes = None

    return {
        'name': img_path.name,
        'path': str(img_path.resolve()),
        'label_path': str(label_path.resolve()) if label_path.exists() else None,
        'labels': labels,
        'boxes': boxes
    }
