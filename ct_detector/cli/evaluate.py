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

The script outputs results to the console. For multi-model comparison, a table
summarizing metrics across all models is printed.
"""

import argparse

# ----------------------------------------------------------------
# CLI Implementation (Subcommands: 'single' or 'compare')
# ----------------------------------------------------------------

def parse_cli_args():
    """
    Parse CLI arguments to handle two subcommands:
      - single: Evaluate a single model
      - compare: Compare multiple models

    Returns an argparse Namespace where `args.command` is either 'single' or 'compare'.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate or compare YOLO models using Ultralytics' .val()"
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Subcommand: single
    single_parser = subparsers.add_parser('single', help='Evaluate a single YOLO model on a dataset')
    single_parser.add_argument('--model', type=str, required=True, help='Path to YOLO model weights')
    single_parser.add_argument('--data', type=str, required=True, help='Path to data config or dataset')
    single_parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    single_parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold for NMS')
    single_parser.add_argument('--max_det', type=int, default=300, help='Max detections per image')
    single_parser.add_argument('--device', type=str, default='', help="Device: ''=Auto, 'cpu', '0', etc.")
    single_parser.add_argument('--imgsz', type=int, default=640, help='Inference image size')
    single_parser.add_argument('--save_json', action='store_true', help='Save output to JSON (COCO format)')
    single_parser.add_argument('--project', type=str, default='runs/val', help='Project path for results')
    single_parser.add_argument('--name', type=str, default='exp', help='Subfolder under project path')
    single_parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    # Subcommand: compare
    compare_parser = subparsers.add_parser('compare', help='Compare multiple YOLO models on the same dataset')
    compare_parser.add_argument('--models', nargs='+', required=True, help='Paths to YOLO model weights (space-separated)')
    compare_parser.add_argument('--data', type=str, required=True, help='Path to data config or dataset')
    compare_parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    compare_parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold for NMS')
    compare_parser.add_argument('--max_det', type=int, default=300, help='Max detections per image')
    compare_parser.add_argument('--device', type=str, default='', help="Device: ''=Auto, 'cpu', '0', etc.")
    compare_parser.add_argument('--imgsz', type=int, default=640, help='Inference image size')
    compare_parser.add_argument('--project', type=str, default='runs/val', help='Project path for results')
    compare_parser.add_argument('--name', type=str, default='compare', help='Subfolder under project path')
    compare_parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    return parser.parse_args()


def main():
    """
    Main entry point for CLI usage. Determines which subcommand
    was requested ('single' or 'compare') and calls the appropriate function
    with the parsed arguments.
    """
    args = parse_cli_args()

    if args.command == 'single':
        # Single-model evaluation
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

        # Print out some common metrics at the end (non-verbose summary)
        print(f"\n--- Evaluation Results ---\n")
        print(f"mAP@0.5:        {results.box.map:0.3f}")
        map_50_95 = getattr(results.box, 'map50_95', None)
        if map_50_95:
            print(f"mAP@0.5:0.95:   {map_50_95:0.3f}")
        print(f"Precision:      {results.box.precision:0.3f}")
        print(f"Recall:         {results.box.recall:0.3f}")
        print(f"Inference time: {results.speed['inference']:0.2f} ms/frame")
        print(f"NMS time:       {results.speed['nms']:0.2f} ms/frame\n")

    elif args.command == 'compare':
        # Multi-model comparison
        raw_results = compare_models(
            model_paths=args.models,
            data_path=args.data,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            imgsz=args.imgsz,
            project=args.project,
            name=args.name,
            verbose=args.verbose
        )
        # The compare_models function already prints a table of results.
        # raw_results is a dict: {model_name: results_object}, if further analysis is needed.

    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == '__main__':
    main()