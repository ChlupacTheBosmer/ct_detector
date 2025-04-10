import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Callable, Union, Generator, Any, Dict, Tuple
from ct_detector.display import COLOR_CONVERSIONS, DEFAULT_CONVERSION
import shutil
from typing import List


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

    Returns a list of images as [H x W x 3] BGR arrays and a list of paths.
    """
    if isinstance(source, np.ndarray):
        # Single image array
        if source.ndim == 3:
            return [source], ["image_01.nparray"]
        else:
            raise ValueError("NumPy array must be shape (H, W, C) for a single image.")
    elif isinstance(source, list) and all(isinstance(item, np.ndarray) for item in source):
        # List of arrays
        return source, [f"image_{i:02d}.nparray" for i in range(len(source))]

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
            return [img], [str(source_path)]
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
            return [img], [str(source_path)]
    else:
        # It's a directory => load all images inside
        return _load_images_from_dir(source_path, extensions)


def _load_images_from_txt(txt_path: Path, extensions: List[str]) -> List[np.ndarray]:
    """
    Reads each line from a .txt file as a path (absolute or relative),
    loads each image in BGR format, and returns in a list.
    """
    images = []
    paths = []
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
        paths.append(str(img_path))
    return images, paths


def _load_images_from_dir(dir_path: Path, extensions: List[str]) -> List[np.ndarray]:
    """
    Scans a directory for recognized image files, loads them, returns a list of BGR arrays.
    """
    images = []
    paths = []
    for file in sorted(dir_path.iterdir()):
        if file.suffix.lower() in extensions:
            img = cv2.imread(str(file))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if img is None:
                raise ValueError(f"Failed to load image: {file}")
            images.append(img)
            paths.append(str(file))
    return images, paths


from PIL import Image
import numpy as np


# def load_image(path: Union[str, Path]) -> Tuple[np.ndarray, Tuple[int, int]]:
#     """
#     Load image from path using PIL and convert to NumPy array.
#
#     Returns:
#         img (np.ndarray): Image array (HWC, uint8).
#         shape (tuple): Image height and width.
#     """
#     path = str(path)
#     with Image.open(path) as im:
#         im = im.convert("RGB")
#         img = np.array(im)
#     return img, img.shape[:2]  # (H, W)

import cv2

# def load_image(path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
#     """
#     Load image using OpenCV and convert to RGB.
#
#     Returns:
#         img (np.ndarray): Image array (HWC, uint8), RGB format.
#         shape (tuple): Image height and width.
#     """
#     bgr = cv2.imread(path, cv2.IMREAD_COLOR)  # Always loads as BGR
#     if bgr is None:
#         raise FileNotFoundError(f"Could not read image: {path}")
#     rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
#     return bgr, bgr.shape[:2]

def load_image(image_path, conversion=None):
    import cv2

    def get_color_conversion(conversion):
        if conversion is None:
            conv = DEFAULT_CONVERSION
        else:
            conv = conversion.upper()
            if conv not in COLOR_CONVERSIONS:
                conv = DEFAULT_CONVERSION
        return COLOR_CONVERSIONS[conv]

    def imread_unicode(path):
        stream = np.fromfile(path, dtype=np.uint8)  # This handles Unicode
        image = cv2.imdecode(stream, cv2.IMREAD_UNCHANGED)
        return image

    image, shape = None, None

    try:
        image = imread_unicode(image_path)
        shape = image.shape[:2]
    except Exception as e:
        stream = open(image_path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        image = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        shape = image.shape[:2]
    finally:

        color_code = get_color_conversion(conversion)

        if color_code is not None:
            try:
                image = cv2.cvtColor(image, color_code)
            except Exception as e:
                pass

        return image, shape


def copy_images(source_dir: str, output_dir: str, file_names: List[str]) -> None:
    """
    Recursively search the source directory for specific image files and copy them into the output directory.

    If an image file already exists in the output directory, the function compares the sizes:
        - If the sizes are the same, the file is assumed to be identical and is skipped.
        - If the sizes differ, the new file is copied with a suffix appended to its base name.
          The suffix is the name of the immediate parent folder from which the file was found.

    Note:
        The file_names list may contain names with or without file extensions.
        If an extension is provided, an exact (case-insensitive) match is performed.
        Otherwise, a match is made on the file's stem (name without extension).

    Args:
        source_dir (str): The root directory to search for the image files (search is recursive).
        output_dir (str): The directory where the image files will be copied.
        file_names (List[str]): A list of file names to be copied (with or without extensions).
    """
    # Create Path objects for easier manipulation and ensure output directory exists.
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Separate names with extensions from those without, using lowercase for case-insensitive matching.
    names_with_ext = set()
    names_without_ext = set()
    for name in file_names:
        if Path(name).suffix:  # if an extension is present
            names_with_ext.add(name.lower())
        else:
            names_without_ext.add(name.lower())

    # Walk through the source directory recursively
    for root, _, files in os.walk(source_path):
        for file in files:
            file_lower = file.lower()
            file_path = Path(root) / file

            # Determine if the file matches one of the specified file names.
            # Check for an exact match if the user provided an extension.
            match = False
            if file_lower in names_with_ext:
                match = True
            else:
                # If no extension was provided in the search term,
                # compare only the base name (stem) of the file.
                if file_path.stem.lower() in names_without_ext:
                    match = True

            if not match:
                continue  # Skip files that do not match

            # Define the target file path in the output directory.
            target_file = output_path / file

            # If a file with the same name already exists in the output directory…
            if target_file.exists():
                # Compare the file sizes to decide whether to skip or rename.
                existing_size = target_file.stat().st_size
                new_size = file_path.stat().st_size

                if existing_size == new_size:
                    # Files are assumed to be identical; do not copy again.
                    print(f"Skipping {file_path} (identical file exists)")
                    continue
                else:
                    # Files differ; add a suffix to the new file name.
                    # The suffix is the immediate parent folder name where the file was found.
                    parent_folder = file_path.parent.name
                    new_name = f"{file_path.stem}_{parent_folder}{file_path.suffix}"
                    target_file = output_path / new_name
                    # If the target filename with the suffix also exists and differs,
                    # append a counter until an unused name is found.
                    counter = 1
                    while target_file.exists() and target_file.stat().st_size != new_size:
                        new_name = f"{file_path.stem}_{parent_folder}_{counter}{file_path.suffix}"
                        target_file = output_path / new_name
                        counter += 1

            # Copy the file (using copy2 to preserve metadata)
            shutil.copy2(file_path, target_file)
            print(f"Copied {file_path} to {target_file}")