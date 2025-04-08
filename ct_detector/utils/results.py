import torch
import numpy as np
from typing import List, Optional, Dict
from ultralytics.engine.results import Results, Boxes


def combine_boxes(boxes1: Boxes, boxes2: Boxes) -> Boxes:
    """
    Combine two Boxes instances into one by concatenating their bounding box data.

    :param boxes1: First Boxes instance. Must have the same original image shape as boxes2.
    :param boxes2: Second Boxes instance. Must have the same original image shape as boxes1.
    :return: A new Boxes instance containing all bounding boxes from both inputs.
    :raises AssertionError: If the original shapes differ.
    :raises TypeError: If data is not a torch.Tensor or numpy.ndarray.
    """
    assert boxes1.orig_shape == boxes2.orig_shape, "Boxes must have the same original image shape"

    if isinstance(boxes1.data, torch.Tensor):
        combined_data = torch.cat([boxes1.data, boxes2.data], dim=0)
    elif isinstance(boxes1.data, np.ndarray):
        combined_data = np.concatenate([boxes1.data, boxes2.data], axis=0)
    else:
        raise TypeError("Unsupported data type for Boxes")

    return Boxes(combined_data, boxes1.orig_shape)


def nms(
    results_list: List[Results],
    iou_thres: float = 0.5,
    conf_thres: float = 0.0,
    class_agnostic: bool = True,
    class_merge_map: Optional[Dict[int, int]] = None
) -> Optional[Results]:
    """
    Custom NMS to merge bounding boxes from multiple models for the SAME frame.

    Steps:
      1) Concatenate all boxes from each Results into a single [N,6] => (x1, y1, x2, y2, conf, cls). Tracking must happen only after this. Otherwise the format is different.
      2) Discard boxes with conf < conf_thres.
      3) Sort remaining boxes by confidence desc.
      4) Loop-based NMS:
         - If class_agnostic=True, all boxes can overlap-suppress each other.
         - Else if class_merge_map is provided, two boxes can overlap-suppress each other if they share the same group ID.
         - Else, they must share the EXACT same cls ID.
         - IoU >= iou_thres => the lower-confidence box is suppressed.
      5) Return a single merged Results with final set of boxes assigned to the first Results item.

    Args:
        results_list: One Results object per model for the same frame.
        iou_thres: IoU threshold for merging overlap.
        conf_thres: Confidence threshold below which boxes are removed.
        class_agnostic: If True, merges across all classes.
                        If False, merges only if classes match (or share a group in class_merge_map).
        class_merge_map: A dictionary mapping original cls ID -> group ID.
                         If two classes share the same group, they can suppress each other.
                         E.g., {0:0, 2:0, 1:1} means classes 0 and 2 are the same group, class 1 is separate.
                         If None, we don't do any class grouping beyond normal logic.

    Returns:
        A single merged Results object (with .boxes updated) or None if no input.
    """
    if not results_list:
        return None

    # We'll copy metadata (orig_shape, etc.) from the first results
    base = results_list[0]

    # 1) Combine all boxes from all models
    combined_boxes_list = []
    for r in results_list:
        if r.boxes is not None and len(r.boxes) > 0:
            combined_boxes_list.append(r.boxes.data)
    if not combined_boxes_list:
        # No boxes => just return base
        return base

    boxes_all = torch.cat(combined_boxes_list, dim=0)  # shape [N,6]

    # 2) Filter out boxes below conf_thres
    if conf_thres > 0.0:
        mask = boxes_all[:, 4] >= conf_thres
        boxes_all = boxes_all[mask]
        if len(boxes_all) == 0:
            return base

    # 3) Sort by confidence descending
    _, idxs = boxes_all[:, 4].sort(descending=True)
    boxes_all = boxes_all[idxs]  # shape [N,6] in descending conf

    # 4) NMS loop
    # We'll keep selected boxes in a list
    selected_boxes = []

    def compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union = areaA + areaB - interArea
        return interArea / union if union > 0 else 0.0

    for i in range(len(boxes_all)):
        candidate = boxes_all[i]  # shape [6]
        cand_cls = int(candidate[5].item())

        keep = True
        for sb in selected_boxes:
            sb_cls = int(sb[5].item())

            # figure out if we should consider them the "same class or group"
            if class_agnostic:
                # always consider overlap
                pass
            else:
                # If we have a class_merge_map, let's see if both classes map to the same group
                if class_merge_map is not None:
                    # If cand_cls or sb_cls not in map, we treat them as separate groups
                    cand_grp = class_merge_map.get(cand_cls, cand_cls)
                    sb_grp   = class_merge_map.get(sb_cls, sb_cls)
                    if cand_grp != sb_grp:
                        # different group => skip overlap check
                        continue
                else:
                    # if no map => must match exact class ID
                    if cand_cls != sb_cls:
                        continue

            # if we get here => we consider them the same group => check IoU
            overlap = compute_iou(candidate[:4], sb[:4])
            if overlap >= iou_thres:
                keep = False
                break

        if keep:
            selected_boxes.append(candidate)

    if not selected_boxes:
        # no final boxes => return base with no boxes
        base.boxes = Boxes(torch.empty((0,6)), base.orig_shape)
        return base

    final_tensor = torch.stack(selected_boxes, dim=0)  # shape [K,6]
    base.boxes = Boxes(final_tensor, base.orig_shape)
    return base


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - interArea
    return interArea / union if union > 0 else 0.0


def unpack_box(box):
    if len(box) == 7:
        # If the box has an ID, it's a tracked box
        x1, y1, x2, y2, track_id, c_conf, c_cls = box
    else:
        x1, y1, x2, y2, c_conf, c_cls = box
        track_id = -1  # No tracking ID
    c_cls = int(c_cls)
    track_id = int(track_id)

    return x1, y1, x2, y2, track_id, c_conf, c_cls