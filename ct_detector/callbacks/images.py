import os
import cv2
import math
import numpy as np
import torch
from pathlib import Path
from typing import Optional, List
from PIL import Image  # for final saving if needed
# If needed, ensure you have: from ultralytics.utils.plotting import save_one_box
# or the correct import path to the function in your local environment

from ultralytics.utils.plotting import save_one_box
from ultralytics.engine.results import Results

from ct_detector.callbacks.base import predict_callback
from ct_detector.callbacks.database import CLASS_MAP


# 4) Helper function to create the mosaic
def create_mosaic(crop_list: List[np.ndarray], mosaic_size=640) -> Optional[np.ndarray]:
    if not crop_list:
        return None

    n = len(crop_list)
    grid_cols = int(math.ceil(math.sqrt(n)))
    grid_rows = int(math.ceil(n / grid_cols))
    tile_w = mosaic_size // grid_cols
    tile_h = mosaic_size // grid_rows

    # create blank mosaic
    mosaic_img = np.zeros((grid_rows * tile_h, grid_cols * tile_w, 3), dtype=np.uint8)

    idx = 0
    for row in range(grid_rows):
        for col in range(grid_cols):
            if idx >= n:
                break
            crop = crop_list[idx]
            h, w, _ = crop.shape

            # Compute scale to fill in the tile
            scale = min(tile_w / w, tile_h / h)

            # Now resize using the final scale
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

            y_off = row * tile_h
            x_off = col * tile_w
            mosaic_img[y_off:y_off + new_h, x_off:x_off + new_w] = resized
            idx += 1

    return mosaic_img


def save_cls_mosaic(
        output_dir: str = "mosaic_output",
        mosaic_size: int = 640,
        padding: int = 10,
        class_map: dict = CLASS_MAP,
        suffix: str = ""
):
    """
    Creates a callback function that, for each image (Results), generates:
      - Ears mosaic (all boxes with class=2),
      - Tusks mosaic (class=3),
      - Elephant/calf mosaic (classes=0 or 1).

    Each mosaic is stored in separate folder:
      output_dir/ears_mosaics/, output_dir/tusks_mosaics/, output_dir/elephants_mosaics/.

    The mosaic is up to 'mosaic_size' in each dimension. We do a grid-based layout,
    resizing each crop if needed to fit in a NxN mosaic.

    usage:
        cb = mosaic_callback(output_dir="mosaics", mosaic_size=640, padding=10)
        # pass cb to your pipeline as a frame_callback or post-inference callback

    The callback expects signature (results: Results, frame_idx: int).

    Implementation details:
    - We rely on 'save_one_box' from Ultralytics to extract and return a square crop of each bounding box,
      with 'square=True, gain=1.02, pad=<padding>, BGR=True, save=False'.
    - We accumulate these crops for each class group (ear, tusk, elephant/calf), build a mosaic,
      and write the mosaic images to disk.
    - If any error or missing data, we skip silently for that image.
    """

    @predict_callback
    def _callback(results: List[Results]):
        # Make subfolders
        ears_dir = os.path.join(output_dir, "ears_mosaics")
        tusks_dir = os.path.join(output_dir, "tusks_mosaics")
        elephants_dir = os.path.join(output_dir, "elephants_mosaics")
        for d in [ears_dir, tusks_dir, elephants_dir]:
            os.makedirs(d, exist_ok=True)

        for r in results:
            # 1) Load original image
            if hasattr(r, 'orig_img') and r.orig_img is not None:
                # 'orig_img' is typically BGR from YOLO pipeline
                orig_bgr = r.orig_img
            elif hasattr(r, 'path') and r.path is not None:
                # read from disk
                img_path = r.path
                if isinstance(img_path, list):
                    # in case it's a list => take the first
                    img_path = img_path[0]

                img_path = Path(img_path)
                if img_path.is_file():
                    orig_bgr = cv2.imread(str(img_path))
                    if orig_bgr is None:
                        print(f"[mosaic_callback] Could not read image: {img_path}")
                        return
                else:
                    print(f"[mosaic_callback] Image path is not a file: {img_path}")
                    return
            else:
                print("[mosaic_callback] No valid original image available in results.")
                return

            # 2) Prepare lists for ear crops, tusk crops, elephant/calf crops
            ear_crops = []
            tusk_crops = []
            ele_crops = []  # includes calves

            # 3) If we have bounding boxes, go through them
            if r.boxes is not None and len(r.boxes) > 0:
                # shape => [N,6] => x1,y1,x2,y2, conf, cls
                for box in r.boxes.data.cpu().numpy():
                    x1, y1, x2, y2, conf, cls = box
                    cls = int(cls)

                    # 'save_one_box' expects a 1D or [1,4] shape in xyxy format
                    # We'll convert to a torch tensor. Then square, pad, etc.
                    xyxy = torch.tensor([x1, y1, x2, y2], dtype=torch.float32).unsqueeze(0)

                    try:
                        # BGR=True => keep color channels in BGR order for cv2 usage
                        crop_img = save_one_box(
                            xyxy,
                            orig_bgr,
                            file=Path("unused.jpg"),  # We'll pass a dummy path since save=False
                            gain=1.02,
                            pad=padding,
                            square=True,
                            BGR=True,
                            save=False
                        )
                        # crop_img is now a NumPy array with shape (H, W, 3)
                    except Exception as e:
                        print(f"[mosaic_callback] Could not crop box: {e}")
                        continue

                    if cls == class_map['ear']:  # ear
                        ear_crops.append(crop_img)
                    elif cls == class_map['tusk']:  # tusk
                        tusk_crops.append(crop_img)
                    elif cls in [class_map['elephant'], class_map['calf']]:  # elephant or calf
                        ele_crops.append(crop_img)
                    else:
                        # skip unknown classes
                        pass

            # 5) Create each mosaic
            ear_mosaic = create_mosaic(ear_crops, mosaic_size=mosaic_size)
            tusk_mosaic = create_mosaic(tusk_crops, mosaic_size=mosaic_size)
            ele_mosaic = create_mosaic(ele_crops, mosaic_size=mosaic_size)

            # 6) Determine base filename
            if isinstance(r.path, list):
                base_name = Path(r.path[0]).stem
            else:
                base_name = Path(str(r.path)).stem

            # 7) Save mosaic images if they exist
            if ear_mosaic is not None:
                ear_path = os.path.join(ears_dir, f"{base_name}{suffix}_ears.jpg")
                cv2.imwrite(ear_path, ear_mosaic)

            if tusk_mosaic is not None:
                tusk_path = os.path.join(tusks_dir, f"{base_name}{suffix}_tusks.jpg")
                cv2.imwrite(tusk_path, tusk_mosaic)

            if ele_mosaic is not None:
                ele_path = os.path.join(elephants_dir, f"{base_name}{suffix}_elephants.jpg")
                cv2.imwrite(ele_path, ele_mosaic)

    return _callback