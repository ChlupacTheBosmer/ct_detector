#!/usr/bin/env python3

"""
predict.py

A script that allows you to either:
1) Perform single-model predictions on a set of images (using CtPredictor).
2) Perform ensemble predictions with multiple YOLO models on the same images (using CtEnsembler).

This script uses the Ultralytics YOLO models under the hood, but exposes
a simpler CLI for specifying:
- The model paths,
- The source of your images (folder, single file, .txt with paths, etc.),
- Confidence thresholding,
- Optional label handling (skip, rename, overwrite),
- Class-agnostic or partially merged class logic for the NMS step,
- and more.

Usage (CLI):

    # SINGLE MODEL:
    python predict.py single \
        --model best.pt \
        --source /path/to/images_or_val.txt \
        --conf 0.25 \
        --save \
        --save_txt \
        --line_width 2 \
        --show_conf \
        --show_labels \
        --handle_labels rename \
        --labels_dir my_custom_labels

    Explanation of these single-model args:
      --model             Path to a single YOLO .pt weight file.
      --source            Input images: directory, single file, or .txt with image paths.
      --conf              Confidence threshold for filtering boxes in the predictor (default=0.25).
      --save              Whether to save annotated images or not.
      --save_txt          Whether to save bounding-box label files.
      --line_width        Pixel width of bounding-box lines in the annotated images.
      --show_conf         Draws confidence values on bounding boxes if set.
      --show_labels       Draws class labels on bounding boxes if set.
      --handle_labels     One of 'skip','rename','overwrite' for existing label files.
      --labels_dir        Custom folder to store .txt labels. Defaults to YOLO's runs/predict/exp/labels.

    # MULTI-MODEL ENSEMBLE:
    python predict.py ensemble \
        --models bestA.pt bestB.pt ... \
        --source /path/to/images \
        --conf 0.25 \
        --nms_iou_thres 0.5 \
        --nms_conf_thres 0.0 \
        --class_agnostic \
        --class_merge_map '{"0":0,"2":0,"1":1}' \
        --max_workers 4

    Explanation of these ensemble args:
      --models            Space-separated list of YOLO .pt weight files to ensemble.
      --source            Input images or .txt file specifying images (same as single-model).
      --conf              Confidence threshold for each single-model predictor.
      --nms_iou_thres     IoU threshold for the final ensemble NMS merging (default=0.5).
      --nms_conf_thres    Confidence threshold for the final ensemble NMS step (default=0.0).
      --class_agnostic    If set, merges bounding boxes across all classes when overlapping.
      --class_merge_map   JSON dict specifying partial merges (e.g. Elephant & Calf share a group).
      --max_workers       Number of threads to run concurrency for the ensemble (default=4).

For additional details or customization, consult the script code, which
integrates with CtPredictor (for single-model detection) or CtDetectionEnsembler
(for multi-model detection) to handle concurrency, bounding-box merging
with custom NMS, and label file management.
"""

import argparse
import sys
from pathlib import Path

# Import your custom classes
from ultralytics import YOLO
from ct_detector.model.predict import CtPredictor
from ct_detector.model.ensemble import CtEnsembler

# ----------------------------------------------------------------
# CLI Implementation (Subcommands: 'single' or 'ensemble')
# ----------------------------------------------------------------

def parse_args():
    """
    Builds an argparse parser with two subcommands:
      - single: Run a single-model prediction with CtPredictor
      - ensemble: Run multiple-model detection with CtEnsembler
    """
    parser = argparse.ArgumentParser(description="Simple CLI for single or ensemble YOLO predictions.")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # -------------------------
    # single subcommand
    # -------------------------
    single_parser = subparsers.add_parser('single', help='Run single-model prediction.')
    single_parser.add_argument('--model', type=str, required=True,
                               help='Path to a single .pt YOLO model weight.')
    single_parser.add_argument('--source', type=str, required=True,
                               help='Source of images (folder, single file, .txt of paths, etc.).')
    single_parser.add_argument('--conf', type=float, default=0.25,
                               help='Confidence threshold passed to predictor (default=0.25).')
    single_parser.add_argument('--handle_labels', type=str, default='overwrite',
                               choices=['skip','rename','overwrite'],
                               help="How to handle existing label files (skip, rename, or overwrite). (default=overwrite)")
    single_parser.add_argument('--labels_dir', type=str, default='',
                               help="Custom directory to store .txt label files (if saving). Default is '' => use YOLO default runs/predict/exp/labels.")
    single_parser.add_argument('--save', action='store_true',
                               help='If set, saves annotated images in the YOLO style output folder.')
    single_parser.add_argument('--save_txt', action='store_true',
                               help='If set, saves bounding box labels to .txt files.')
    single_parser.add_argument('--line_width', type=int, default=None,
                               help='Bounding box line width for visualization.')
    single_parser.add_argument('--show_conf', action='store_true',
                               help='If set, displays confidence on the bounding boxes.')
    single_parser.add_argument('--show_labels', action='store_true',
                               help='If set, displays class labels on the bounding boxes.')
    # etc. => Add or remove any extra single-model arguments

    # -------------------------
    # ensemble subcommand
    # -------------------------
    ensemble_parser = subparsers.add_parser('ensemble', help='Run multi-model ensemble detection.')
    ensemble_parser.add_argument('--models', nargs='+', required=True,
                                 help="List of .pt YOLO model paths for the ensemble.")
    ensemble_parser.add_argument('--source', type=str, required=True,
                                 help='Same as in single: folder, single file, .txt, etc.')
    ensemble_parser.add_argument('--conf', type=float, default=0.25,
                                 help='Confidence threshold assigned to each single-model predictor. (default=0.25)')
    ensemble_parser.add_argument('--nms_iou_thres', type=float, default=0.5,
                                 help='IoU threshold used in the final ensemble NMS merging.')
    ensemble_parser.add_argument('--nms_conf_thres', type=float, default=0.0,
                                 help='Confidence threshold used in the final ensemble NMS merging.')
    ensemble_parser.add_argument('--class_agnostic', action='store_true',
                                 help='If set, merges boxes across all classes in NMS.')
    ensemble_parser.add_argument('--class_merge_map', type=str, default='',
                                 help='Optional JSON dict for partially merging classes. E.g. {"0":0,"2":0,"1":1}.')
    ensemble_parser.add_argument('--max_workers', type=int, default=4,
                                 help='Thread count for concurrency across models. (default=4)')
    # etc. => Additional ensemble-specific arguments if needed

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == 'single':
        # ==============
        # SINGLE-MODEL
        # ==============
        overrides = {
            'conf': args.conf,
            'save': args.save,
            'save_txt': args.save_txt,
            'line_width': args.line_width,
            'show_conf': args.show_conf,
            'show_labels': args.show_labels
            # add more if you want
        }
        predictor = CtPredictor(
            overrides=overrides,
            handle_existing_labels=args.handle_labels,
            labels_dir=args.labels_dir
        )

        # Set up YOLO with our custom predictor
        model = YOLO(args.model)
        model.predictor = predictor

        # Now run the predictions
        results = model.predict(source=args.source, stream=False)
        print(f"\n[CLI] Single-model detection done, total images: {len(results)}.")
        for i, r in enumerate(results):
            print(f"Image {i}: {len(r.boxes)} boxes detected.")

    elif args.command == 'ensemble':
        # ==============
        # ENSEMBLE
        # ==============
        # Possibly parse the class_merge_map from JSON if user passed a string
        import json
        if args.class_merge_map:
            class_merge_map = json.loads(args.class_merge_map)
            # convert keys from strings to int
            class_merge_map = {int(k): v for k, v in class_merge_map.items()}
        else:
            class_merge_map = None

        # We create an ensemble object
        from ct_detector.model.ensemble import CtEnsembler
        overrides = {
            'conf': args.conf,
            # if you want to apply e.g. 'save_txt': True for each single-model, etc.
        }
        ensembler = CtEnsembler(
            model_paths=args.models,
            predictor_overrides=overrides
        )

        # define a quick callback if you want
        def frame_callback(merged_res, frame_idx):
            print(f"[CLI Ensemble] Frame {frame_idx}: {len(merged_res.boxes)} boxes")

        # run the predictions
        gen = ensembler.predict(
            source=args.source,
            nms_iou_thres=args.nms_iou_thres,
            nms_conf_thres=args.nms_conf_thres,
            class_agnostic=args.class_agnostic,
            class_merge_map=class_merge_map,
            frame_callback=frame_callback,
            max_workers=args.max_workers
        )

        # Just exhaust the generator so it runs
        count = 0
        for merged_frame in gen:
            count += 1
        print(f"\n[CLI] Ensemble detection done, total frames processed: {count}.")

    else:
        print("Unknown command:", args.command)
        sys.exit(1)


if __name__ == "__main__":
    main()
