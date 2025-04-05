import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Callable, Union, Generator, Any, Dict


def load_images_from_source(
        source: Union[str, Path, np.ndarray, List[np.ndarray]],
        extensions: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".tif"]
) -> List[np.ndarray]:
    """
    Utility function to load images into a list of NumPy arrays (BGR).
    Supported input:
      - Single NumPy array (shape=HWC).
      - List of NumPy arrays.
      - Single file path to an image.
      - Single directory path of images.
      - Single .txt file with lines of image paths.

    Returns a list of images as [H x W x 3] BGR arrays.
    """
    if isinstance(source, np.ndarray):
        # Single image array
        if source.ndim == 3:
            return [source]
        else:
            raise ValueError("NumPy array must be shape (H, W, C) for a single image.")
    elif isinstance(source, list) and all(isinstance(item, np.ndarray) for item in source):
        # List of arrays
        return source

    # Otherwise, source is presumably a path
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")

    if source_path.is_file():
        if source_path.suffix.lower() in extensions:
            # Single image file
            img = cv2.imread(str(source_path))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if img is None:
                raise ValueError(f"Failed to load image: {source_path}")
            return [img]
        elif source_path.suffix.lower() == ".txt":
            # This is a text file of image paths
            return _load_images_from_txt(source_path, extensions)
        else:
            # Possibly a single image not recognized or an unrecognized format
            # We'll try to read it anyway
            img = cv2.imread(str(source_path))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if img is None:
                raise ValueError(f"Failed to load file: {source_path}")
            return [img]
    else:
        # It's a directory => load all images inside
        return _load_images_from_dir(source_path, extensions)


def _load_images_from_txt(txt_path: Path, extensions: List[str]) -> List[np.ndarray]:
    """
    Reads each line from a .txt file as a path (absolute or relative),
    loads each image in BGR format, and returns in a list.
    """
    images = []
    base_dir = txt_path.parent  # so relative paths are resolved
    with open(txt_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    for ln in lines:
        img_path = base_dir / ln  # resolve relative
        if not img_path.exists():
            raise FileNotFoundError(f"Image path does not exist: {img_path}")
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        images.append(img)
    return images


def _load_images_from_dir(dir_path: Path, extensions: List[str]) -> List[np.ndarray]:
    """
    Scans a directory for recognized image files, loads them, returns a list of BGR arrays.
    """
    images = []
    for file in sorted(dir_path.iterdir()):
        if file.suffix.lower() in extensions:
            img = cv2.imread(str(file))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if img is None:
                raise ValueError(f"Failed to load image: {file}")
            images.append(img)
    return images