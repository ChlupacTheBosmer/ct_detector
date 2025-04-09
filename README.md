
# ct_detector: Comprehensive YOLO-based Camera Trap Detection Package

🐘📷🌿

## Overview

`ct_detector` is a Python package specifically designed for running YOLO object detection models on camera trap imagery, primarily focused on detecting elephants and related classes: elephants, calves, ears, and tusks (class IDs 0-3 respectively). The package provides lightweight, easy-to-use tools tailored to the needs of ecologists, significantly expediting the process of sorting through extensive camera trap data that frequently contains many false positives. It includes model management tools and tracking algorithms that not only simplify wildlife research workflows but also offer a practical starting point for roughly estimating individual animals within image sequences. Although initially focused on elephants, `ct_detector` can be easily extended to other data domains, facilitating the detection of various animals and specific body parts critical for individual identification.

🔍🐾🖥️

## Key Functionalities

### Object Detection and Prediction

🐆📸🎯

- **`CtPredictor`**: Custom YOLO-based predictor that facilitates running object detection predictions similar to the standard YOLO predictor. It provides additional flexibility through detailed control over existing labels (skip, rename, overwrite) and directing output label files to custom directories. Moreover, it supports introducing callbacks at various stages of the prediction process, enabling extensive customization such as automated annotation generation, result tracking, and other tailored post-processing tasks.

### Ensemble Predictions

📊🤝🐾

- **`CtEnsembler`**: Facilitates running ensemble predictions with multiple YOLO models concurrently on the same input images, merging results with Non-Maximum Suppression (NMS).

### Model Evaluation

🧪📈📏

- **Evaluation Utilities**: Evaluate single or multiple YOLO models, calculating metrics such as mAP@0.5, mAP@0.5:0.95, precision, recall, and inference speeds.

### Object Tracking

📍🐘👀

- **`CtTracker`**: Supports integration of BYTETracker and BOTSORT trackers with detection results to maintain consistent object IDs across image sequences.

### Dataset Management

📁🗂️🐾

- **`Dataset` Class**: A YOLO-compatible dataset manager capable of loading, manipulating, and splitting datasets from various formats (directories, YAML configs, TXT files).

### Database Integration

🗃️💻📊

- Utilities to analyze bounding box predictions and log comprehensive detection data into SQLite databases, facilitating downstream analysis.

### Image Processing and Mosaics

🖼️🎨🐘

- Create mosaics of detected object classes (ears, tusks, elephants/calves) for visual inspection and individual identification tasks.

### Callbacks and Annotation

⚙️✏️🔄

- Modular callbacks for real-time processing, annotation saving, results visualization, and integration into custom prediction pipelines. By writing custom callbacks, users can significantly expand the package's functionality, tailoring it precisely to their specific needs such as auto-annotating, tracking, and extensive data post-processing.

📚🔖🛠️

## Package Structure

### Core Modules:

- `model/predict.py`: Contains `CtPredictor`, extending YOLO's DetectionPredictor.
- `model/ensemble.py`: `CtEnsembler` class for ensemble predictions.
- `model/track.py`: Implements object tracking (`CtTracker`) and associated utilities.
- `data/dataset.py`: Comprehensive management of datasets using Ultralytics' standards.
- `utils/`: Includes utility scripts for file handling, result parsing, and image loading.
- `callbacks/`: Includes standardized callback functions for result processing, database logging, and image visualization.
- `evaluate.py`: Provides CLI and Python-based evaluation utilities for YOLO models.

### Jupyter Notebooks:

📓💡🐾

All practical examples and demonstrations can be found in the following notebooks:

- `dataset.ipynb`: Example workflow for dataset creation, loading, and splitting.
- `predict.ipynb`: Demonstrates how to use `CtPredictor` for individual image predictions.
- `evaluate.ipynb`: Guides on evaluating models individually and comparatively.
- `deploy.ipynb`: Instructions for deploying models in real-world scenarios.
- `callbacks.ipynb`: Illustrates using callback functions for annotation and database logging.
- `utils.ipynb`: General usage examples of provided utility functions.

## Available YOLO Models

🐾📐🔍

The package supports YOLO models designed to detect elephants and related classes (0: elephant, 1: calf, 2: ear, 3: tusk). Models vary by YOLO versions (YOLOv8, YOLOv9, YOLOv11, YOLOv12, YOLOv8-world) and sizes (nano, small, medium, large). Model files (`.pt`) are provided separately and integrated into your detection pipeline via the provided utilities.

## Test Datasets

🧩📂🔬

- The package contains a few test datasets specific to the data domain, useful for evaluating models or testing prediction workflows.
- Test sets facilitate quick verification of model accuracy and pipeline functionality.

## Usage Examples

📖🔧🐾

### Predicting with CtPredictor:

```python
from ct_detector.model.predict import CtPredictor

predictor = CtPredictor(overrides={'conf': 0.25})
results = predictor.predict(source='path/to/images')
```

### Ensemble Predictions:

```python
from ct_detector.model.ensemble import CtEnsembler

def my_callback(merged_result, frame_idx):
    print(f"Frame {frame_idx}: {len(merged_result.boxes)} boxes")

ensembler = CtEnsembler(model_paths=["modelA.pt", "modelB.pt"])
for result in ensembler.predict("path/to/images", frame_callback=my_callback):
    # Process results
```

### Model Evaluation:

```python
from evaluate import evaluate_model, compare_models

# Single-model evaluation
evaluate_model(model_path='yolov8n.pt', data_path='dataset.yaml')

# Multi-model comparison
compare_models(model_paths=['yolov8n.pt', 'yolov8s.pt'], data_path='dataset.yaml')
```

### Tracking Results:

```python
from ct_detector.model.track import CtTracker, track_results
tracker = CtTracker(tracker_type="botsort.yaml")
callback = track_results(tracker)

# Integrate callback in prediction pipeline
```

## Contributing and Support

🤝🌐🔧

Feel free to contribute via pull requests or raise issues on GitHub for support.

## GitHub

🐙💻📍

Find this project on GitHub: [ct_detector](https://github.com/ChlupacTheBosmer/ct_detector)

---

🌟📸🐾 Enjoy simplifying your camera trap image analysis workflows!
