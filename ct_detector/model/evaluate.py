#!/usr/bin/env python3

"""
evaluate.py

A script that allows you to either:
1) Evaluate a single YOLO model on a dataset.
2) Compare multiple YOLO models side-by-side on the same dataset.

It uses the Ultralytics YOLO package's `.val()` method to compute metrics
like mAP@0.5, mAP@0.5:0.95, precision, and recall. It also gathers speed
(inference and NMS/postprocess times).

Usage (CLI):

    # SINGLE MODEL EVALUATION:
    python evaluate.py single \
        --model yolov8n.pt \
        --data coco128.yaml \
        --conf 0.25 \
        --iou 0.5 \
        --max_det 300 \
        --device '' \
        --imgsz 640 \
        --save_json \
        --project runs/val \
        --name exp \
        --verbose

    # MULTI-MODEL COMPARISON:
    python evaluate.py compare \
        --models yolov8n.pt yolov8s.pt \
        --data coco128.yaml \
        --conf 0.25 \
        --iou 0.5 \
        --max_det 300 \
        --device '' \
        --imgsz 640 \
        --project runs/val \
        --name compare \
        --verbose

Usage (in a Python script or Jupyter notebook):
    from evaluate import evaluate_model, compare_models

    # Single-model evaluation
    results = evaluate_model(model_path='yolov8n.pt', data_path='coco128.yaml')

    # Multi-model comparison
    raw_results_dict = compare_models(
        model_paths=['yolov8n.pt', 'yolov8s.pt'],
        data_path='coco128.yaml'
    )

The script outputs results to the console. For multi-model comparison, a table
summarizing metrics across all models is printed.
"""

from ultralytics import YOLO
import pandas as pd
import os


# ----------------------------------------------------------------
# Helper / Internal Functions
# ----------------------------------------------------------------

def _get_model_metrics(results, model_path):
    """
       Extract key metrics from a model's `results` object and return them as
       a list ready to be placed into a DataFrame row.

       :param results: Ultralytics results object after running .val().
       :param model_path: Path to the model weights file used (string).
       :return: List of string-formatted metric values in a specific order.
   """
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
    """
        Print a DataFrame in a tabular format using 'tabulate' if available.
        Falls back to basic DataFrame printing if tabulate is not installed.

        :param df: A pandas DataFrame with columns and data to display.
    """
    try:
        from tabulate import tabulate
        print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
    except ImportError:
        print("Module 'tabulate' is not available. Please install it to format tables nicely.")
        print(df.to_string(index=False))

def get_metrics(results, model_path):
    """
        Create a brief DataFrame summarizing top-line metrics like fitness, mAP, etc.,
        for a single model's results, then return that DataFrame.

        :param results: Ultralytics results object after .val().
        :param model_path: Path to model weights used for clarity in the table.
        :return: pandas DataFrame with columns for relevant metrics.
    """
    print("\n--- Evaluation Metrics ---")
    df = pd.DataFrame(
        columns=['Model:', 'fitness', 'mAP50(B)', 'mAP50-95(B)', "precision(B)", "recall(B)", "Speed(inference)",
                 "Speed(NMS)"])
    df.loc[len(df)] = _get_model_metrics(results, model_path)

    return df


def get_class_metrics(results):
    """
        Create a DataFrame of per-class metrics: AP, precision, recall.
        This draws from the 'names' (class labels) and 'box.maps', 'box.p', 'box.r' in the results object.

        :param results: Ultralytics results object.
        :return: pandas DataFrame with rows per class and columns for AP, precision, recall.
    """
    cls_maps = {cls: {"map": mp, "p": p, "r": r} for cls, mp, p, r in
                zip(results.names.values(), results.box.maps, results.box.p, results.box.r)}
    print("--- Per-class metrics ---")
    df = pd.DataFrame(columns=['Class:', 'MAP', 'Precision', 'Recall'])
    for k, v in cls_maps.items():
        df.loc[len(df)] = [k, f"{v['map']:.4f}", f"{v['p']:.4f}", f"{v['r']:.4f}"]

    return df


# ----------------------------------------------------------------
# Core Functions for Model Evaluation & Comparison
# ----------------------------------------------------------------


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
    Evaluate a single YOLO model on a specified dataset using Ultralytics' .val() method.

    :param model_path: Path to the YOLO model weights (e.g., 'yolov8n.pt').
    :param data_path: Path to the dataset config .yaml or directory of images.
    :param conf: Confidence threshold for predictions (float).
    :param iou: IoU threshold for NMS (float).
    :param max_det: Maximum number of detections per image (int).
    :param device: Device to run on ('cpu', '0', etc.). '' uses auto-detection.
    :param imgsz: Image size used during validation inference.
    :param save_json: If True, saves results in COCO JSON format (if dataset supports it).
    :param project: Folder path for saving validation outputs (str).
    :param name: Subfolder name within 'project' for storing outputs (str).
    :param verbose: If True, prints extra logs and summary tables.

    :return: A Ultralytics results object containing metrics, including:
             - results.box.map (mAP@0.5)
             - results.box.maps (list of per-class APs)
             - results.results_dict (dictionary of additional metrics)
             - results.speed (dict with speed info: inference, postprocess, etc.)
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
    Compare multiple YOLO models side-by-side on the same dataset. Each model is evaluated
    with the same parameters, then a table of metrics is displayed.

    :param model_paths: List of paths to model weights (e.g., ['modelA.pt', 'modelB.pt']).
    :param data_path: Path to dataset config .yaml or folder of images.
    :param conf: Confidence threshold for predictions.
    :param iou: IoU threshold for NMS.
    :param max_det: Max detections per image.
    :param device: Compute device: ''=auto, 'cpu', '0' for GPU 0, etc.
    :param imgsz: Image size during inference.
    :param project: Base directory for saving evaluation results.
    :param name: Subfolder name under 'project' for storing results.
    :param verbose: If True, prints logs and individual results for each model.

    :return: A dictionary mapping {model_name: results_object} so you can further analyze
             each model's results if needed.
    """

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
        df.loc[len(df)] = _get_model_metrics(results, model_path)

    # Print it nicely in console:
    print("\n--- Model Comparison ---")
    print_metrics_table(df)

    return raw_results


