from functools import wraps
from typing import Union, Callable, Optional, List
from PIL import Image

from ultralytics.engine.results import Results
from ultralytics.engine.predictor import BasePredictor

from ct_detector.display.plots import display_pil_image


def predict_callback(func: Callable[[Results], None]):
    """
    Decorator for standardizing callback functions that operate on Results or BasePredictor objects.
    Ensures the wrapped function always receives a Results and BasePredictor instance.

    :param func: The user-defined function accepting (Results)
    """

    @wraps(func)
    def wrapper(arg: Union[List[Results], Results, BasePredictor]):
        if isinstance(arg, Results):
            results = [arg]
            predictor = None
        elif isinstance(arg, list) and all(isinstance(item, Results) for item in arg):
            results = arg
            predictor = None
        elif isinstance(arg, BasePredictor):
            predictor = arg
            results = predictor.results
        else:
            raise TypeError("Argument must be either a Results or BasePredictor instance.")

        return func(results)

    return wrapper


@predict_callback
def demo_predict_callback(results: List[Results]):
    """
    Example callback function that processes the prediction results.
    This function will be called with either a Results or BasePredictor instance.

    :param results: The prediction results.
    """
    # Process the results here
    print(f"Processing {len(results)} results.")


@predict_callback
def display_results(results: List[Results]):
    """
    Example callback function that displays the prediction results.
    This function will be called with either a Results or BasePredictor instance.

    :param results: The prediction results.
    """
    for r in results:
        display_pil_image(image=Image.fromarray(r.plot()), width=400, height=400)
