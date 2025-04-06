import os
from functools import wraps
from typing import Union, Callable, List

from ultralytics.engine.results import Results
from ultralytics.engine.predictor import BasePredictor

from ct_detector.callbacks.base import predict_callback


def annotate(directory: str, verbose: bool = True):
    """
    Creates a callback function that saves prediction results as .txt files into the specified directory.

    For each Results object in the list, it checks whether:
      - The object is indeed a Results instance.
      - The object has the 'path' attribute.
    It then extracts the base file name (handling multiple dots precisely) and saves the .txt file
    using the result.save_txt method (without saving confidence values).

    :param directory: The directory where .txt files will be saved.
    :param verbose: If True, prints the path of each saved .txt file.
    :return: A callback function that processes a list of Results.
    """

    @predict_callback
    def _callback(results: List[Results]):
        # Create the target directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        for r in results:
            # Validate the necessary attribute
            if not hasattr(r, "path"):
                print(f"Warning: The result object {r} does not have a 'path' attribute.")
                continue

            # Extract the image file name robustly (handling multiple dots)
            base_name = os.path.basename(r.path)  # e.g., "my.image.name.jpg"
            name_without_ext, _ = os.path.splitext(base_name)  # e.g., "my.image.name"
            txt_file = os.path.join(directory, name_without_ext + ".txt")

            try:
                # Save the text file without confidence values
                r.save_txt(txt_file, save_conf=False)
                if verbose:
                    print(f"Saved .txt file: {txt_file}")
            except Exception as e:
                print(f"Error saving .txt file for {r.path}: {e}")

    return _callback