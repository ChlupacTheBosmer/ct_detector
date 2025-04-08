import os
import cv2
import numpy as np
import concurrent.futures
from pathlib import Path
from typing import List, Optional, Callable, Union, Generator, Any, Dict

from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes

from ct_detector.model.predict import CtPredictor  # or your single-model predictor
from ct_detector.utils.results import nms
from ct_detector.utils.files import load_images_from_source


class CtEnsembler:
    """
    A class that runs an ensemble of YOLO-based models (with single-model
    predictors) on a set of images, frame by frame (one-by-one).
    Each frame is run concurrently across all models, then merged immediately,
    and optionally a callback is invoked.

    Example usage:
        ensembler = CtDetectionEnsembler(
            model_paths=["modelA.pt", "modelB.pt"],
            user_predictors=None,  # or a list of custom single-model predictors
            predictor_overrides={"conf":0.25, "save_txt":False} # etc.
        )

        def my_callback(merged_result, frame_idx):
            print(f"Frame {frame_idx} => {len(merged_result.boxes)} boxes")

        # run predictions on a directory or .txt of images
        for merged in ensembler.predict("path/to/dir", frame_callback=my_callback):
            # do logic, e.g. merged.save_txt() if you want final annotation
            ...
    """

    def __init__(
            self,
            model_paths: List[str],
            user_predictors: Optional[List[CtPredictor]] = None,
            predictor_overrides: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            model_paths: List of YOLO model weight paths.
            user_predictors: If provided, must match len(model_paths).
                             Each predictor should handle single-frame inputs.
            predictor_overrides: If creating new single-model predictors, pass these overrides
                                 (e.g., stream=False, batch=1, conf=0.25, etc.).
        """
        if user_predictors and len(user_predictors) != len(model_paths):
            raise ValueError("user_predictors must match len(model_paths) if provided.")

        self.model_paths = model_paths
        self.user_predictors = user_predictors
        self.predictor_overrides = predictor_overrides or {}

        # Initialize YOLO models
        self.models = []
        for i, mp in enumerate(model_paths):
            yolo_model = YOLO(mp)
            if user_predictors:
                # attach the user predictor
                pred = user_predictors[i]
            else:
                # create a new MySingleModelPredictor with the overrides
                pred = CtPredictor(overrides=self.predictor_overrides)
            yolo_model.predictor = pred
            self.models.append(yolo_model)

    def _run_model_on_frame(
            self, model: YOLO, frame: np.ndarray, frame_idx: int
    ) -> Optional[Results]:
        """
        Concurrency-friendly method to run a single model on a single frame.

        Returns a single Results object or None if inference fails.
        """
        try:
            # We'll do model.predict() on this single frame => returns a list of Results
            # If batch=1, typically you get exactly one Results. We'll just return the first.
            results_list = model.predict(frame, stream=False, conf=model.predictor.args.conf)  # no streaming, single-image
            if len(results_list) > 0:
                return results_list[0]
        except Exception as e:
            print(f"Error in model {model.model}, frame {frame_idx}: {e}")
        return None

    def predict(
            self,
            source: Union[str, Path, np.ndarray, List[np.ndarray]],
            merge_fn: Callable[[List[Results]], Results] = nms,
            nms_iou_thres: float = 0.5,
            nms_conf_thres: float = 0.0,
            class_agnostic: bool = True,
            class_merge_map: Optional[Dict[int, int]] = None,
            _callbacks: List[Optional[Callable[[Results], None]]] = None,
            max_workers: int = 4,
    ) -> Generator[Results, None, None]:
        """
        Processes the 'source' one frame at a time. For each frame:
          - runs concurrency for each model
          - merges results
          - calls optional callback
          - yields the merged result

        Args:
            source: Single or multiple images in array form, or a path (dir, single file, .txt of paths).
            merge_fn: Merge function that merges multiple model predictions for the same frame.
            nms_iou_thres: IoU threshold for that merge function (if it uses it).
            nms_conf_thres: Confidence threshold for the merge function (if it uses it).
            class_agnostic: If True, merges across all classes.
            class_merge_map: A dictionary mapping original cls ID -> group ID.
            _callbacks: List of callables optionally called after merging each frame. Is passed merged_result.
            max_workers: concurrency thread count.

        Yields:
            Merged Results object for each frame.
        """
        # 1) Load frames as a list of BGR NumPy arrays
        frames, paths = load_images_from_source(source)

        # 2) For each frame, run each model concurrently
        for frame_idx, frame in enumerate(frames):
            if frame.ndim != 3:
                raise ValueError(f"Invalid frame dimension {frame.ndim}. Expected 3D array (H,W,C).")

            results_this_frame = []
            # concurrency
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_model = {
                    executor.submit(self._run_model_on_frame, m, frame, frame_idx): m
                    for m in self.models
                }
                for fut in concurrent.futures.as_completed(future_to_model):
                    model_obj = future_to_model[fut]
                    try:
                        res = fut.result()
                        if res is not None:
                            results_this_frame.append(res)
                    except Exception as e:
                        print(f"Ensemble concurrency error on frame {frame_idx}: {e}")

            # 3) Merge the model predictions for this frame
            if results_this_frame:
                merged = merge_fn(results_this_frame,
                                  iou_thres=nms_iou_thres,
                                  conf_thres=nms_conf_thres,
                                  class_agnostic=class_agnostic,
                                  class_merge_map=class_merge_map
                                  )
                if merged is not None:

                    # Attach metadata to the merged result
                    merged.orig_img = frame
                    merged.save_dir = results_this_frame[0].save_dir if hasattr(results_this_frame[0], 'save_dir') else None
                    merged.names = results_this_frame[0].names if hasattr(results_this_frame[0], 'names') else None
                    merged.path = paths[frame_idx] if paths and len(
                        paths) > frame_idx else None  # attach the path if available

                    predictor = CtPredictor()
                    predictor.results = [merged]

                    # 4) Callbacks
                    if _callbacks and isinstance(_callbacks, list) and len(_callbacks) > 0:
                        for cb in _callbacks:
                            if callable(cb):
                                try:
                                    cb(predictor)
                                except Exception as e:
                                    print(f"Frame callback error: {e}")
                            else:
                                print(f"Callback {cb} is not callable")

                    yield merged
