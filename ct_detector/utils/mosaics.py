import os
from PIL import Image
import math
import numpy as np
import cv2
from typing import List, Optional


def create_mosaic(images: List[np.ndarray], mosaic_size: Optional[int] = None,
                  tile_size: Optional[int] = None, tiles_per_row: Optional[int] = None) -> Optional[
    np.ndarray]:
    """
    Create a mosaic from a list of images (NumPy arrays).

    Args:
        images (List[np.ndarray]): List of images as NumPy arrays to be placed in the mosaic.
        mosaic_size (Optional[int]): The maximum size of the mosaic (both width and height) in pixels.
        tile_size (Optional[int]): The size of each tile (in pixels).
        tiles_per_row (Optional[int]): The number of tiles per row in the mosaic.

    Returns:
        Optional[np.ndarray]: The mosaic image as a NumPy array or None if the list is empty.
    """

    if not images:
        return None

    n = len(images)

    # Determine mosaic layout (grid size) based on the parameters
    if mosaic_size:
        grid_cols = int(math.ceil(math.sqrt(n)))
        grid_rows = int(math.ceil(n / grid_cols))
        tile_w = mosaic_size // grid_cols
        tile_h = mosaic_size // grid_rows
    elif tile_size and tiles_per_row:
        grid_cols = tiles_per_row
        grid_rows = int(math.ceil(n / grid_cols))
        tile_w = tile_h = tile_size
    else:
        raise ValueError("You must specify either 'mosaic_size' or both 'tile_size' and 'tiles_per_row'.")

    # Create the blank mosaic image
    mosaic_img = np.zeros((grid_rows * tile_h, grid_cols * tile_w, 3), dtype=np.uint8)

    idx = 0
    for row in range(grid_rows):
        for col in range(grid_cols):
            if idx >= n:
                break
            crop = images[idx]
            h, w, _ = crop.shape

            # Compute scale to fit the tile
            scale = min(tile_w / w, tile_h / h)

            # Resize the image to fit the tile dimensions
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Calculate the offset position in the mosaic
            y_off = row * tile_h
            x_off = col * tile_w
            mosaic_img[y_off:y_off + new_h, x_off:x_off + new_w] = resized
            idx += 1

    return mosaic_img


def create_mosaic_from_folder(subdir, output_size=320, max_size_multiplier=5):
    """
    Process a single subfolder: resize images and create mosaics using `create_mosaic_from_images`.
    """
    # Create the mosaics folder within the subfolder
    mosaics_folder = os.path.join(subdir, 'mosaics')
    os.makedirs(mosaics_folder, exist_ok=True)

    # Get the name of the subfolder to use in mosaic filenames
    folder_name = os.path.basename(subdir)

    # Filter and collect image files (assume image files end with .jpg, .png, etc.)
    files = os.listdir(subdir)
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print(f"No images found in {subdir}. Skipping...")
        return

    # Resize images to the specified size and collect them
    resized_images = []
    for image_file in image_files:
        img_path = os.path.join(subdir, image_file)
        img = Image.open(img_path)

        # Ensure the image is square, taking the larger dimension if needed
        max_dim = max(img.size)
        img = img.resize((max_dim, max_dim))

        # Resize the image to the specified output_size
        img_resized = img.resize((output_size, output_size))
        resized_images.append(np.array(img_resized))

    # Create the mosaic using the new function
    mosaic_img = create_mosaic(
        images=resized_images,
        mosaic_size=output_size * max_size_multiplier
    )

    if mosaic_img is not None:

        # Check if filename exists in the location and if yes add a number to it, check if that exists too and if so increment the number until it doesn't
        count = 1
        mosaic_filename = f"{folder_name}_mosaic_{count}.png"
        while os.path.exists(os.path.join(mosaics_folder, mosaic_filename)):
            mosaic_filename = f"{folder_name}_mosaic_{count}.png"
            count += 1

        # Save the mosaic image
        mosaic_img = Image.fromarray(mosaic_img)
        mosaic_img.save(os.path.join(mosaics_folder, mosaic_filename))
        print(f"Created mosaic: {mosaic_filename}")


def create_mosaics_from_folders(input_folder, output_size=320, max_size_multiplier=5):
    """
    Iterate over all nested folders and process them using process_subfolder.
    """
    # Traverse through all subfolders in the input folder
    for subdir, _, _ in os.walk(input_folder):
        # Only process nested folders (skip the main folder itself)
        if subdir == input_folder:
            continue

        # Process each subfolder
        create_mosaic_from_folder(subdir, output_size, max_size_multiplier)