#!/usr/bin/env python3

"""
evaluate.py

A script providing:
1) A function to evaluate a single YOLO model using the Ultralytics .val() method.
2) A function to compare multiple YOLO models side-by-side and display results as a table.

Usage (CLI for single model):
    python evaluate.py --model yolov8n.pt --data coco128.yaml --conf 0.25 --iou 0.5

Usage (notebook or Python script):
    from evaluate import evaluate_yolo_model, compare_models

    # Single model:
    results = evaluate_yolo_model(...)

    # Multiple models:
    compare_models(
        model_paths=['modelA.pt', 'modelB.pt'],
        data_path='coco128.yaml'
    )
"""

import argparse
from ultralytics import YOLO
import pandas as pd
import os
# Optional: If you want to format tables nicely in console, you can install tabulate

def _get_model_metrics(results, model_path):
    return [
        os.path.basename(os.path.splitext(model_path)[0]),
        f"{results.results_dict['fitness']:.4f}",
        f"{results.results_dict['metrics/mAP50(B)']:.4f}",
        f"{results.results_dict['metrics/mAP50-95(B)']:.4f}",
        f"{results.results_dict['metrics/precision(B)']:.4f}",
        f"{results.results_dict['metrics/recall(B)']:.4f}",
        f"{results.speed['inference']:.3f} ms/frame",
        f"{results.speed['postprocess']:.3f} ms/frame"
    ]

def print_metrics_table(df):
    try:
        from tabulate import tabulate
        print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
    except ImportError:
        print("Module 'tabulate' is not available. Please install it to format tables nicely.")
        print(df.to_string(index=False))

def get_metrics(results, model_path):
    print("\n--- Evaluation Metrics ---")
    df = pd.DataFrame(
        columns=['Model:', 'fitness', 'mAP50(B)', 'mAP50-95(B)', "precision(B)", "recall(B)", "Speed(inference)",
                 "Speed(NMS)"])
    df.loc[len(df)] = _get_model_metrics(results, model_path)

    return df


def get_class_metrics(results):
    cls_maps = {cls: {"map": mp, "p": p, "r": r} for cls, mp, p, r in
                zip(results.names.values(), results.box.maps, results.box.p, results.box.r)}
    print("--- Per-class metrics ---")
    df = pd.DataFrame(columns=['Class:', 'MAP', 'Precision', 'Recall'])
    for k, v in cls_maps.items():
        df.loc[len(df)] = [k, f"{v['map']:.4f}", f"{v['p']:.4f}", f"{v['r']:.4f}"]

    return df


def evaluate_model(
        model_path: str,
        data_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
        device: str = '',  # '' = auto, 'cpu', or '0' for GPU 0, etc.
        imgsz: int = 640,
        save_json: bool = False,
        project: str = 'runs/val',
        name: str = 'exp',
        verbose: bool = False
):
    """
    Evaluate a YOLO model using Ultralytics' .val() method.

    :param model_path: Path to the YOLO model weights (e.g., 'yolov8n.pt').
    :param data_path: Path to the dataset config .yaml or directory of images.
    :param conf: Confidence threshold for inference.
    :param iou: IoU threshold for NMS.
    :param max_det: Maximum number of detections per image.
    :param device: Device to run on ('cpu', '0', etc.). '' = auto.
    :param imgsz: Image size during inference/validation.
    :param save_json: If True, save results to JSON (COCO format).
    :param project: Where to save evaluation results.
    :param name: Subfolder name for results inside 'project'.
    :param verbose: If True, print additional logs.

    :return: A Ultralytics 'Metrics' object with evaluation results,
             including fields like box.map (mAP@0.5), box.maps (per-class AP), speed, etc.
    """
    # Load the model
    model = YOLO(model_path)

    # Run evaluation
    results = model.val(
        data=data_path,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=device,
        imgsz=imgsz,
        save_json=save_json,
        project=project,
        name=name,
        verbose=verbose
    )

    if verbose:
        # Print the evaluation metrics
        print_metrics_table(get_metrics(results, model_path))
        print_metrics_table(get_class_metrics(results))

    return results


def compare_models(
        model_paths,
        data_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
        device: str = '',
        imgsz: int = 640,
        project: str = 'runs/val',
        name: str = 'compare',
        verbose: bool = False
):
    """
    Evaluate multiple YOLO models on the same dataset and display their metrics side-by-side.

    :param model_paths: List of paths to model weights (e.g., ['modelA.pt', 'modelB.pt']).
    :param data_path: Path to dataset config .yaml or folder of images.
    :param conf: Confidence threshold.
    :param iou: IoU threshold for NMS.
    :param max_det: Max detections per image.
    :param device: Compute device: ''=auto, 'cpu', '0' for GPU.
    :param imgsz: Inference image size.
    :param verbose: If True, print extra info.
    """
    import pandas as pd

    # Prepare a DataFrame to store results
    df = pd.DataFrame(
        columns=['Model:', 'fitness', 'mAP50(B)', 'mAP50-95(B)', "precision(B)", "recall(B)", "Speed(inference)",
                 "Speed(NMS)"])

    # Prepare dictionary to store raw results
    raw_results = {}

    for idx, model_path in enumerate(model_paths):
        # We can create a name suffix like "name_idx" to keep results from each model separate
        name_suffix = f"{name}_{idx}"

        if verbose:
            print(f"\nEvaluating model: {model_path}")

        # Evaluate the model
        results = evaluate_model(
            model_path=model_path,
            data_path=data_path,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
            imgsz=imgsz,
            project=project,
            name=name_suffix,
            verbose=verbose
        )

        # Store the results in a dictionary for later use
        raw_results[os.path.basename(os.path.splitext(model_path)[0])] = results

        # Collect relevant metrics
        # For YOLOv8, results.box.map is mAP@0.5
        # results.box.maps is a list of per-class APs
        # COCO-style mAP@0.5:0.95 is often reported as results.box.map50_95 in older versions,
        # but in recent versions, it's stored in results.box.map if you run multiple IoU thresholds.
        # We'll use results.box.map for 0.5, and results.box.map50_95 if available, or fallback to the same if not.
        # Additional metrics from results.box are results.box.precision, results.box.recall, etc.

        df.loc[len(df)] = _get_model_metrics(results, model_path)

    #     # The "map" attribute is mAP@0.5 by default in the latest ultralytics package
    #     # The "map50_95" attribute is the COCO style
    #     map_05 = getattr(results.box, 'map50', None) or 0.0
    #     map_05_95 = getattr(results.box, 'map', None)  # might be None if not computed
    #     if map_05_95 is None:
    #         # If it's not available, we just reuse map_05 to avoid None
    #         map_05_95 = map_05
    #
    #     precision = getattr(results.box, 'mp', 0.0)
    #     recall = getattr(results.box, 'mr', 0.0)
    #
    #     # Speed info in results.speed is a dict: {'preprocess': x, 'inference': y, 'nms': z}
    #     inference_ms = results.speed.get('inference', 0.0)
    #     nms_ms = results.speed.get('postprocess', 0.0)
    #
    #     comparison_data["Model"].append(model_path)
    #     comparison_data["mAP@0.5"].append(map_05)
    #     comparison_data["mAP@0.5:0.95"].append(map_05_95)
    #     comparison_data["Precision"].append(precision)
    #     comparison_data["Recall"].append(recall)
    #     comparison_data["Inference (ms/img)"].append(inference_ms)
    #     comparison_data["NMS (ms/img)"].append(nms_ms)
    #
    # # Create a pandas DataFrame
    # df = pd.DataFrame(comparison_data)

    # Print it nicely in console:
    print("\n--- Model Comparison ---")
    print_metrics_table(df)

    return raw_results


def parse_args():
    """Parse command-line arguments for single-model evaluation (not comparison)."""
    parser = argparse.ArgumentParser(description="Evaluate a YOLO model using Ultralytics' .val()")

    parser.add_argument('--model', type=str, required=True, help='Path to YOLO model weights')
    parser.add_argument('--data', type=str, required=True, help='Path to data config or dataset')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold for NMS')
    parser.add_argument('--max_det', type=int, default=300, help='Maximum detections per image')
    parser.add_argument('--device', type=str, default='', help="Device: ''=Auto, 'cpu', '0', etc.")
    parser.add_argument('--imgsz', type=int, default=640, help='Inference image size')
    parser.add_argument('--save_json', action='store_true', help='Save output to JSON (COCO format)')
    parser.add_argument('--project', type=str, default='runs/val', help='Project path for results')
    parser.add_argument('--name', type=str, default='exp', help='Subfolder name under project path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    return parser.parse_args()


def main():
    # This main() is for single-model eval usage from CLI
    args = parse_args()

    # Evaluate
    results = evaluate_model(
        model_path=args.model,
        data_path=args.data,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        imgsz=args.imgsz,
        save_json=args.save_json,
        project=args.project,
        name=args.name,
        verbose=args.verbose
    )

    # Print out some common metrics
    print(f"\n--- Evaluation Results ---\n")
    print(f"mAP@0.5: {results.box.map:0.3f}")
    # The 'map50_95' might be there if you have multi-IoU evaluation
    map_50_95 = getattr(results.box, 'map50_95', None)
    if map_50_95:
        print(f"mAP@0.5:0.95: {map_50_95:0.3f}")

    print(f"Precision: {results.box.precision:0.3f}")
    print(f"Recall: {results.box.recall:0.3f}")
    print(f"Inference speed: {results.speed['inference']:0.2f} ms/frame")
    print(f"NMS speed: {results.speed['nms']:0.2f} ms/frame\n")


if __name__ == '__main__':
    main()
