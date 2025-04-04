from ultralytics import settings
from ct_detector.model import DATASETS_DIR, MODELS_DIR, MODELS
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Update YOLO native settings file to look for datasets and store results in custom dirs
settings.update({'datasets_dir': DATASETS_DIR, 'runs_dir': os.path.join(ROOT_DIR, 'runs')})

__all__ = [ROOT_DIR, 'DATASETS_DIR', 'MODELS_DIR', 'MODELS']