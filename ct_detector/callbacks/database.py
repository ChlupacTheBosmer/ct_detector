import sqlite3
import json
import math
from types import SimpleNamespace
from typing import List, Optional, Callable, Union, Generator, Any, Dict

from ultralytics.engine.results import Results, Boxes

from ct_detector.callbacks.base import predict_callback
from ct_detector.utils.results import iou

CLASS_MAP = {
    "calf": 0,
    "ear": 1,
    "elephant": 2,
    "tusk": 3
}


def analyze_boxes(boxes: Boxes, class_map: dict = CLASS_MAP):
    """
    Analyze the bounding boxes and extract relevant data about made predictions.

    :param boxes: Boxes object containing bounding box data.
    :param class_map: Mapping class names to their respective indices.
    :return: A SimpleNamespace object containing various statistics about the predictions.
    """
    data_dict = {
        "boxes_data": [],
        "conf_values": [],
        "num_elephants": 0,
        "num_calves": 0,
        "ears_visible": 0,
        "tusks_visible": 0,
        "max_ear_diag": 0.0,
        "max_tusk_diag": 0.0,
        "ear_to_body_ratio": 0.0,
        "tusk_to_body_ratio": 0.0,
        "elephant_diag_for_largest_ear": 0.0,
        "elephant_diag_for_largest_tusk": 0.0
    }

    data = SimpleNamespace(**data_dict)

    # We'll keep track of bounding boxes in a pythonic list
    # shape [N,6] => x1,y1,x2,y2, conf, cls
    if boxes is not None and len(boxes) > 0:
        all_boxes = boxes.data.cpu().numpy()
        # Example: for box in all_boxes => [x1,y1,x2,y2, conf, cls]
        for box in all_boxes:
            x1, y1, x2, y2, c_conf, c_cls = box
            c_cls = int(c_cls)

            # store for JSON logging
            data.boxes_data.append([float(x1), float(y1), float(x2), float(y2), float(c_conf), c_cls])
            data.conf_values.append(float(c_conf))

            # check class
            diag = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if c_cls == class_map['elephant']:
                data.num_elephants += 1
            elif c_cls == class_map['calf']:
                data.num_calves += 1
            elif c_cls == class_map['ear']:
                data.ears_visible += 1
                if diag > data.max_ear_diag:
                    data.max_ear_diag = diag
            elif c_cls == class_map['tusk']:
                data.tusks_visible += 1
                if diag > data.max_tusk_diag:
                    data.max_tusk_diag = diag
            else:
                # unknown class => ignore or handle
                pass

    def get_max_cls_box(boxes: Boxes, cls: int):
        max_box = None
        max_diag = 0.0
        if boxes is not None and len(boxes) > 0:
            for box in boxes.data.cpu().numpy():
                x1, y1, x2, y2, c_conf, c_cls = box
                if int(c_cls) == cls:
                    diag = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if diag >= max_diag:
                        max_diag = diag
                        max_box = box
        return max_box, max_diag

    def get_max_iou_cls_box(boxes: Boxes, ref_box: list, cls: int):
        best_iou = 0.0
        best_box = None
        best_box_diag = 0.0
        if ref_box is not None:
            ref_xyxy = ref_box[:4]
            for box in boxes.data.cpu().numpy():
                x1, y1, x2, y2, c_conf, c_cls = box
                if int(c_cls) == cls:
                    overlap = iou(ref_box[:4], box[:4])
                    if overlap > best_iou:
                        best_iou = overlap
                        best_box_diag = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return best_box, best_box_diag

    # Get the largest ear box and diagonal
    max_ear_box, max_ear_diag = get_max_cls_box(boxes, class_map['ear'])

    # Get the elephant box and its diagonal that has the largest overlap with the largest ear
    best_ele_box, best_ele_diag = get_max_iou_cls_box(boxes, max_ear_box, class_map['elephant'])

    # If we have the right elephant box, we can calculate the ratio between the largest ear and its elephant
    data.elephant_diag_for_largest_ear = best_ele_diag
    if data.elephant_diag_for_largest_ear > 0.0:
        data.ear_to_body_ratio = data.max_ear_diag / data.elephant_diag_for_largest_ear

    # Get the largest tusk box and diagonal
    max_tusk_box, max_tusk_diag = get_max_cls_box(boxes, class_map['tusk'])

    # Get the elephant box and its diagonal that has the largest overlap with the largest tusk
    best_ele_box, best_ele_diag = get_max_iou_cls_box(boxes, max_tusk_box, class_map['elephant'])

    # If we have the right elephant box, we can calculate the ratio between the largest tusk and its elephant
    data.elephant_diag_for_largest_tusk = best_ele_diag
    if data.elephant_diag_for_largest_tusk > 0.0:
        data.tusk_to_body_ratio = data.max_tusk_diag / data.elephant_diag_for_largest_tusk

    return data


@predict_callback
def print_prediction_data(results: List[Results]):
    """
    Callback function to print prediction data for each image.
    This function is used to test logging the results of the detection pipeline.

    :param results:
    :return:
    """

    for r in results:

        # 2) Parse data from results
        image_id = r.path  # or just Path(results.path).name
        if isinstance(image_id, (list, tuple)):
            # if your pipeline yields multiple images batch => pick e.g. the first.
            # Usually results.path is a single string
            image_id = str(image_id[0])
        else:
            image_id = str(image_id)

        # 3) Analyze boxes and extract data
        data = analyze_boxes(r.boxes, CLASS_MAP)

        data.image_id = image_id

        # Format and print all attributes of the data namespace
        for attr, value in data.__dict__.items():
            if isinstance(value, (list, tuple)):
                print(f"{attr}: {json.dumps(value)}")
            else:
                print(f"{attr}: {value}")


def log_prediction_data_into_sqlite(
        db_path: str,
        table_name: str = "data",
        create_if_missing: bool = True,
        class_map: dict = CLASS_MAP
):
    """
    Creates a callback function that logs detection results into an SQLite database.

    Usage:
        callback = sqlite_db_callback(db_path="my_database.sqlite", table_name="my_table")
        # Then pass 'callback' to your pipeline as a frame_callback or post_inference callback.

    The resulting callback function expects two parameters:
        (results: Results, frame_idx: int)

    For each image:
    1) We parse bounding boxes to fill these columns:
       - image_id (text)
       - num_elephants (int)
       - num_calves (int)
       - ears_visible (int)
       - tusks_visible (int)
       - boxes_data (text) => JSON array of [x1,y1,x2,y2,conf,cls]
       - conf_values (text) => JSON array of confidences
       - max_ear_diagonal (float)
       - max_tusk_diagonal (float)
       - max_elephant_diagonal_for_largest_ear (float)
       - max_elephant_diagonal_for_largest_tusk (float)
       - ear_to_body_ratio (float)
       - tusk_to_body_ratio (float)
       - time_of_day (text)
       - location (text)
       - camera_id (text)

    2) We insert or append a row to the 'elephant_data' (or chosen) table.

    We do robust error handling, so if any row fails to insert,
    we print an error but continue processing.
    """

    # Inner function that actually does the logging
    @predict_callback
    def _callback(results: List[Results]):
        """
        Callback invoked per frame/image. 'results' has bounding boxes in results.boxes,
        plus metadata like results.path, results.orig_shape, etc.
        """
        # We open the DB for each callback. Alternatively, you can open once outside
        # and keep a persistent connection. For streaming usage, opening each time is simpler
        # but less performant. Adjust as needed.
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()

            # Create table if missing
            if create_if_missing:
                create_sql = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    image_id TEXT PRIMARY KEY,
                    num_elephants INTEGER,
                    num_calves INTEGER,
                    ears_visible INTEGER,
                    tusks_visible INTEGER,
                    boxes_data TEXT,
                    conf_values TEXT,
                    max_ear_diagonal REAL,
                    max_tusk_diagonal REAL,
                    elephant_diagonal_for_largest_ear REAL,
                    elephant_diagonal_for_largest_tusk REAL,
                    ear_to_body_ratio REAL,
                    tusk_to_body_ratio REAL,
                    time_of_day TEXT,
                    location TEXT,
                    camera_id TEXT
                );
                """
                cur.execute(create_sql)

            for r in results:
                # 2) Parse data from results
                image_id = r.path  # or just Path(results.path).name
                if isinstance(image_id, (list, tuple)):
                    # if your pipeline yields multiple images batch => pick e.g. the first.
                    # Usually results.path is a single string
                    image_id = str(image_id[0])
                else:
                    image_id = str(image_id)

                # 3) Analyze boxes and extract data
                data = analyze_boxes(r.boxes, class_map)

                # 4) Insert row into DB
                insert_sql = f"""
                INSERT OR REPLACE INTO {table_name} (
                    image_id, num_elephants, num_calves, ears_visible, tusks_visible,
                    boxes_data, conf_values, max_ear_diagonal, max_tusk_diagonal,
                    elephant_diagonal_for_largest_ear, elephant_diagonal_for_largest_tusk, ear_to_body_ratio, 
                    tusk_to_body_ratio, time_of_day, location, camera_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """

                # in future, user can supply time_of_day/location/camera_id from metadata
                # for now, we store them as blank
                db_row = (
                    image_id,
                    data.num_elephants,
                    data.num_calves,
                    data.ears_visible,
                    data.tusks_visible,
                    json.dumps(data.boxes_data),
                    json.dumps(data.conf_values),
                    float(data.max_ear_diag),
                    float(data.max_tusk_diag),
                    float(data.elephant_diag_for_largest_ear),
                    float(data.elephant_diag_for_largest_tusk),
                    float(data.ear_to_body_ratio),
                    float(data.tusk_to_body_ratio),
                    "",  # time_of_day
                    "",  # location
                    ""  # camera_id
                )

                cur.execute(insert_sql, db_row)
                conn.commit()
        except Exception as e:
            # robust error handling => keep going
            print(f"[sqlite_db_callback] Error inserting data for image: {r.path}. Reason: {e}")
        finally:
            if conn:
                conn.close()

    return _callback
