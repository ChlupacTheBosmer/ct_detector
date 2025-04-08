from ct_detector.model.track import CtTracker
from ultralytics.engine.predictor import BasePredictor
from typing import Optional


def track_results(tracker: CtTracker, persist: bool = True, filter: bool = False):
    """
    Track results of the model.

    :param tracker:
    :param persist:
    :param filter:
    :return:
    """
    def _callback(predictor: Optional[BasePredictor]):
        """
        Track results of the model.
        :return:
        """
        for i, r in enumerate(predictor.results):
            try:
                r = tracker.process_tracking(r, persist=persist, filter=filter)
            except Exception as e:
                print(f"Error in tracking: {e}")

            predictor.results[i] = r

    return _callback

