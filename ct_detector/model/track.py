from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.utils.checks import check_yaml
from ultralytics.engine.results import Results, Boxes
from typing import Optional, List
import numpy as np
import torch
import traceback


class CtTracker:
    # A mapping of tracker types to corresponding tracker classes
    TRACKER_MAP = {'bytetrack': BYTETracker, 'botsort': BOTSORT}

    def __init__(self,tracker_type: str = "botsort.yaml", trackers_number: int = 1):
        self.tracker_type = tracker_type
        self.trackers_number = trackers_number
        self.trackers = None
        self.trackers = self._load_trackers()

    def _load_trackers(self, persist: bool = False):
        if self.trackers is not None and persist:
            return
        try:
            tracker = check_yaml(self.tracker_type)
            cfg = IterableSimpleNamespace(**yaml_load(tracker))

            if cfg.tracker_type not in ['bytetrack', 'botsort']:
                raise ValueError(f"Unsupported tracker type: '{cfg.tracker_type}'")

            trackers = []
            for _ in range(self.trackers_number):
                tracker = self.TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
                trackers.append(tracker)

            return trackers
        except Exception as e:
            raise Exception(f"Error in loading trackers: {e}")

    def process_tracking(self,
                         result: Results,
                         tracker_index: int = 0,
                         persist: bool = True,
                         filter: bool = False):

        """
        Process a batch of frames with its detection results and update with object tracking.

        Args:
            result: (list, optional): Results object with detections. Also contains a frame in attr.
            tracker_index (int, optional): The index of the tracker to use. Defaults to 0.
            persist (bool, optional): Whether to persist the tracker if it already exists. Defaults to True.
            filter (bool, optional): Whether to filter the tracked boxes to only include the tracked boxes. Defaults to False.
        """

        if not isinstance(result, Results):
            raise ValueError("Detection result must be an instance of Results")

        if not self.trackers:
            raise Exception("Trackers not initialized")

        # Make sure that the selected tracker index is within the range of the trackers
        tracker_index = max(0, min(tracker_index, len(self.trackers) - 1))

        try:
            if not persist:
                self.trackers[tracker_index].reset()

            if result is None or result.boxes is None or len(result.boxes) == 0:  # Check for None values
                print("No detection")
                return result
            else:
                detection_boxes = result.boxes.cpu().numpy()

            tracks = self.trackers[tracker_index].update(detection_boxes, result.orig_img)

            # print("Boxes:", detection_boxes.data)
            # print("Tracks:", tracks)

            if len(tracks) == 0:
                #print("No tracks")
                return result

            if filter:
                # Get last column of the tracked boxes for the indexes of the tracked boxes in the original data
                idx = tracks[:, -1].astype(int)
                updated_boxes = detection_boxes.data[idx]
            else:
                # Add the untracked boxes to the tracked boxes with value of id -1
                updated_boxes = add_missing_bboxes(detection_boxes.data, tracks)

            result.boxes = Boxes(torch.as_tensor(updated_boxes), result.boxes.orig_shape)

            return result
        except Exception as e:
            traceback.print_exc()
            raise Exception(f"Error in processing tracking: {e}")



def add_missing_bboxes(bboxes: np.ndarray, tracked_bboxes: np.ndarray):
    """
    Find and append missing bounding boxes after tracking.
    :param bboxes: Original bounding boxes with format [x, y, x, y, conf, cls]
    :param tracked_bboxes: Tracked bounding boxes with format [x, y, x, y, track_id, conf, cls, index_in_original_array]
    :return: Resulting bounding boxes with format [x, y, x, y, track_id, conf, cls]
    """
    # Initialize the result array with the same number of rows as the original bboxes and 7 columns
    result_bboxes = np.zeros((bboxes.shape[0], 7))

    # Fill in the tracked boxes in their original positions
    for tracked_bbox in tracked_bboxes:
        idx = int(tracked_bbox[7])  # index in original array
        result_bboxes[idx, :4] = tracked_bbox[:4]  # x, y, x, y
        result_bboxes[idx, 4] = tracked_bbox[4]  # track_id
        result_bboxes[idx, 5] = tracked_bbox[5]  # conf
        result_bboxes[idx, 6] = tracked_bbox[6]  # cls

    # Find the rows in bboxes that are not in tracked_bboxes and add them with track_id = -1
    for i in range(bboxes.shape[0]):
        if result_bboxes[i, 5] == 0 and result_bboxes[i, 2] == 0:  # If the confidence and x2 are still 0, it means this box was not tracked
            result_bboxes[i, :4] = bboxes[i, :4]  # x, y, x, y
            result_bboxes[i, 4] = -1  # track_id
            result_bboxes[i, 5] = bboxes[i, 4]  # conf
            result_bboxes[i, 6] = bboxes[i, 5]  # cls

    return result_bboxes

