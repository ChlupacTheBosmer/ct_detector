import os
from ct_detector.model.evaluate import evaluate_model, compare_models
from ct_detector.model.predict import CtPredictor
from ct_detector.model.ensemble import CtEnsembler

def get_assets_path():
    """
    Get the path to the assets/models directory from the root of the repository.
    :return: Path to the assets/models directory.
    """
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Move up two directories to reach the root
    root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))

    # Construct the path to assets/models
    assets_path = os.path.join(root_dir, "assets")
    return assets_path


# From a given directory get all .pt files and construct a dictionary of them where key is the file name and value is
# the full path to the file
def get_models_from_directory(directory: str) -> dict:
    """
    Get all .pt files from a given directory and construct a dictionary of them.

    :param directory: The directory to search for .pt files.
    :return: A dictionary where the key is the file name and the value is the full path to the file.
    """
    models = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".pt"):
                models[os.path.splitext(file)[0]] = os.path.join(root, file)
    return models

# Get the paths to each .yaml dataset file within the nested folders of the datasets directory. Create a dictionary
# where the key is the dataset name and the value is the full path to the .yaml file. Only include files that end with .yaml.
def get_datasets_from_directory(directory: str) -> dict:
    """
    Get all .yaml files from a given directory and construct a dictionary of them.

    :param directory: The directory to search for .yaml files.
    :return: A dictionary where the key is the file name and the value is the full path to the file.
    """
    datasets = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".yaml"):
                datasets[os.path.splitext(file)[0]] = os.path.join(root, file)
    return datasets

MODELS_DIR = os.path.join(get_assets_path(), "models")
DATASETS_DIR = os.path.join(get_assets_path(), "datasets")

MODELS = get_models_from_directory(MODELS_DIR)
DATASETS = get_datasets_from_directory(DATASETS_DIR)

__all__ = [MODELS,
           evaluate_model,
           compare_models,
           CtPredictor,
           CtEnsembler]