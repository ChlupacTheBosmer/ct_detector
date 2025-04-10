import os.path
import sqlite3
import json
import math
from types import SimpleNamespace
from typing import List, Optional, Callable, Union, Generator, Any, Dict

from ultralytics.engine.results import Results, Boxes

from ct_detector.callbacks.base import predict_callback
from ct_detector.utils.results import iou, unpack_box

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
        "elephant_ids": [],
        "calf_ids": [],
        "num_individuals": 0,
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

        # Example: for box in all_boxes => [x1,y1,x2,y2, (id), conf, cls]
        for box in all_boxes:

            # Unpack the box data
            x1, y1, x2, y2, track_id, c_conf, c_cls = unpack_box(box)

            # store for JSON logging
            data.boxes_data.append([float(x1), float(y1), float(x2), float(y2), track_id, float(c_conf), c_cls])
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

        if boxes.is_track:
            # If the boxes are tracked, we can get the IDs of the elephants and calves
            for box in all_boxes:
                x1, y1, x2, y2, track_id, c_conf, c_cls = unpack_box(box)
                if int(c_cls) == class_map['elephant'] and track_id != -1:
                    data.elephant_ids.append(int(track_id))
                elif int(c_cls) == class_map['calf'] and track_id != -1:
                    data.calf_ids.append(int(track_id))

            # Create a list of all IDs
            all_ids = data.elephant_ids + data.calf_ids
            data.num_individuals = len(set(all_ids))  # unique IDs

    def get_max_cls_box(boxes: Boxes, cls: int):
        max_box = None
        max_diag = 0.0
        if boxes is not None and len(boxes) > 0:
            for box in boxes.data.cpu().numpy():
                x1, y1, x2, y2, track_id, c_conf, c_cls = unpack_box(box)
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
                x1, y1, x2, y2, track_id, c_conf, c_cls = unpack_box(box)
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
        class_map: dict = CLASS_MAP,
        metadata: Optional[Dict[str, Any]] = None
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
       - elephant_ids (text) => JSON array of IDs
       - calf_ids (text) => JSON array of IDs
       - num_individuals (int)
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
                    elephant_ids TEXT,
                    calf_ids TEXT,
                    num_individuals INTEGER,
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
                    image_id, num_elephants, num_calves, elephant_ids,
                    calf_ids, num_individuals, ears_visible, tusks_visible,
                    boxes_data, conf_values, max_ear_diagonal, max_tusk_diagonal,
                    elephant_diagonal_for_largest_ear, elephant_diagonal_for_largest_tusk, ear_to_body_ratio, 
                    tusk_to_body_ratio, time_of_day, location, camera_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """

                # in future, user can supply time_of_day/location/camera_id from metadata
                # for now, we store them as blank
                db_row = (
                    image_id,
                    data.num_elephants,
                    data.num_calves,
                    json.dumps(data.elephant_ids),
                    json.dumps(data.calf_ids),
                    data.num_individuals,
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
                    metadata.get("time_od_day", "") if metadata else "",  # time_of_day
                    metadata.get("location", "") if metadata else "",  # location
                    metadata.get("camera_id", "") if metadata else ""  # camera_id
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


import sqlite3
import json
import numpy as np
from collections import defaultdict


def create_individual_summary_table(conn, table_name):
    """
    Create a table that summarizes the number of elephants, calves, and total detected individuals by location.

    Args:
    conn (sqlite3.Connection): The SQLite connection object.
    table_name (str): The name of the table to create.

    Returns:
    None
    """
    cur = conn.cursor()

    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        location TEXT PRIMARY KEY,
        num_elephants INTEGER,
        num_calves INTEGER,
        num_total INTEGER,
        id_elephants TEXT,
        id_calves TEXT,
        id_total TEXT,
        avg_frames_ele REAL,
        avg_frames_cal REAL,
        avg_frames_all REAL
    );
    """
    cur.execute(create_sql)
    conn.commit()


def create_camera_summary_table(conn, table_name):
    """
    Create a table that summarizes the number of elephants, calves, and total detected individuals by location and camera.

    Args:
    conn (sqlite3.Connection): The SQLite connection object.
    table_name (str): The name of the table to create.

    Returns:
    None
    """
    cur = conn.cursor()

    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        location TEXT,
        camera_id TEXT,
        num_elephants INTEGER,
        num_calves INTEGER,
        num_total INTEGER,
        id_elephants TEXT,
        id_calves TEXT,
        id_total TEXT,
        avg_frames_ele REAL,
        avg_frames_cal REAL,
        avg_frames_all REAL,
        PRIMARY KEY (location, camera_id)
    );
    """
    cur.execute(create_sql)
    conn.commit()


def create_individual_details_table(conn, table_name):
    """
    Create a table that holds detailed information about individual detected IDs.

    Args:
    conn (sqlite3.Connection): The SQLite connection object.
    table_name (str): The name of the table to create.

    Returns:
    None
    """
    cur = conn.cursor()

    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        ID TEXT,
        location TEXT,
        camera_ids TEXT,
        num_frames INTEGER,
        img_names TEXT,
        PRIMARY KEY (ID, location)
    );
    """
    cur.execute(create_sql)
    conn.commit()


def process_individuals_data(conn, source_table, summary_table, camera_summary_table, individual_details_table):
    """
    Process the data from the original table and insert into the three summary tables:
    - Individual summary by location
    - Camera summary by location and camera_id
    - Individual details for each ID

    Args:
    conn (sqlite3.Connection): The SQLite connection object.
    source_table (str): The name of the original table containing the elephant data.
    summary_table (str): The name of the table to store individual summary data by location.
    camera_summary_table (str): The name of the table to store camera-based summary data.
    individual_details_table (str): The name of the table to store individual ID details.

    Returns:
    None
    """
    cur = conn.cursor()

    # Step 1: Fetch all records from the original table
    cur.execute(f"SELECT * FROM {source_table};")
    rows = cur.fetchall()

    # Initialize data structures to hold calculated values
    location_data = defaultdict(lambda: {'elephants': 0, 'calves': 0, 'individuals': set(), 'elephant_ids': set(),
                                         'calf_ids': set(), 'frames_ele': defaultdict(int),
                                         'frames_cal': defaultdict(int), 'frames_all': defaultdict(int),
                                         'camera_ids': set()})
    camera_data = defaultdict(lambda: {'elephants': 0, 'calves': 0, 'individuals': set(), 'elephant_ids': set(),
                                       'calf_ids': set(), 'frames_ele': defaultdict(int),
                                       'frames_cal': defaultdict(int), 'frames_all': defaultdict(int)})

    individual_data = defaultdict(lambda: {'location': set(), 'camera_ids': set(), 'num_frames': 0, 'img_names': set()})

    # Process each row in the source table
    for row in rows:
        image_id, num_elephants, num_calves, elephant_ids, calf_ids, num_individuals, \
            ears_visible, tusks_visible, boxes_data, conf_values, max_ear_diagonal, max_tusk_diagonal, \
            elephant_diagonal_for_largest_ear, elephant_diagonal_for_largest_tusk, ear_to_body_ratio, tusk_to_body_ratio, \
            time_of_day, location, camera_id = row

        # Parse the elephant and calf IDs from the JSON strings
        elephant_ids = set(json.loads(elephant_ids))
        calf_ids = set(json.loads(calf_ids))

        # Update location-level data
        location_data[location]['elephants'] += num_elephants
        location_data[location]['calves'] += num_calves
        location_data[location]['elephant_ids'].update(elephant_ids)
        location_data[location]['calf_ids'].update(calf_ids)
        location_data[location]['individuals'].update(elephant_ids, calf_ids)
        location_data[location]['camera_ids'].add(camera_id)

        # Update camera-level data
        camera_data[(location, camera_id)]['elephants'] += num_elephants
        camera_data[(location, camera_id)]['calves'] += num_calves
        camera_data[(location, camera_id)]['elephant_ids'].update(elephant_ids)
        camera_data[(location, camera_id)]['calf_ids'].update(calf_ids)
        camera_data[(location, camera_id)]['individuals'].update(elephant_ids, calf_ids)

        # Track frames for individuals (per location and camera)
        for ele_id in elephant_ids:
            location_data[location]['frames_ele'][ele_id] += 1
            camera_data[(location, camera_id)]['frames_ele'][ele_id] += 1
        for cal_id in calf_ids:
            location_data[location]['frames_cal'][cal_id] += 1
            camera_data[(location, camera_id)]['frames_cal'][cal_id] += 1
        for ele_id in elephant_ids | calf_ids:
            location_data[location]['frames_all'][ele_id] += 1
            camera_data[(location, camera_id)]['frames_all'][ele_id] += 1
            individual_data[ele_id]['location'].add(location)
            individual_data[ele_id]['camera_ids'].add(camera_id)
            individual_data[ele_id]['img_names'].add(os.path.basename(image_id))

    # Step 2: Insert summary data for each location (aggregated across all cameras)
    for location, data in location_data.items():
        num_elephants = data['elephants']
        num_calves = data['calves']
        num_total = num_elephants + num_calves

        # Only elephant IDs for id_elephants, only calf IDs for id_calves, and both combined for id_total
        id_elephants = json.dumps(list(data['elephant_ids']))
        id_calves = json.dumps(list(data['calf_ids']))
        id_total = json.dumps(list(data['individuals']))

        avg_frames_ele = np.mean(list(data['frames_ele'].values())) if data['frames_ele'] else 0
        avg_frames_cal = np.mean(list(data['frames_cal'].values())) if data['frames_cal'] else 0
        avg_frames_all = np.mean(list(data['frames_all'].values())) if data['frames_all'] else 0

        # Insert into the summary table for location-based data
        cur.execute(f"""
        INSERT OR REPLACE INTO {summary_table} 
        (location, num_elephants, num_calves, num_total, id_elephants, id_calves, id_total, 
        avg_frames_ele, avg_frames_cal, avg_frames_all)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (location, num_elephants, num_calves, num_total, id_elephants, id_calves, id_total,
              avg_frames_ele, avg_frames_cal, avg_frames_all))

    # Step 3: Insert camera-specific data into the camera summary table (distinct per camera_id)
    for (location, camera_id), data in camera_data.items():
        num_elephants = data['elephants']
        num_calves = data['calves']
        num_total = num_elephants + num_calves

        # Only elephant IDs for id_elephants, only calf IDs for id_calves, and both combined for id_total
        id_elephants = json.dumps(list(data['elephant_ids']))
        id_calves = json.dumps(list(data['calf_ids']))
        id_total = json.dumps(list(data['individuals']))

        avg_frames_ele = np.mean(list(data['frames_ele'].values())) if data['frames_ele'] else 0
        avg_frames_cal = np.mean(list(data['frames_cal'].values())) if data['frames_cal'] else 0
        avg_frames_all = np.mean(list(data['frames_all'].values())) if data['frames_all'] else 0

        # Insert into the camera summary table for camera-specific data
        cur.execute(f"""
        INSERT OR REPLACE INTO {camera_summary_table} 
        (location, camera_id, num_elephants, num_calves, num_total, id_elephants, id_calves, id_total, 
        avg_frames_ele, avg_frames_cal, avg_frames_all)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (location, camera_id, num_elephants, num_calves, num_total, id_elephants, id_calves, id_total,
              avg_frames_ele, avg_frames_cal, avg_frames_all))

    # Step 4: Insert individual details into the third table
    for individual_id, data in individual_data.items():
        location = ', '.join(data['location'])  # Join multiple locations if any
        camera_ids = json.dumps(list(data['camera_ids']))
        img_names = json.dumps(list(data['img_names']))
        num_frames = len(data['img_names'])  # Count the number of frames where the ID appears

        # Insert individual details into the table
        cur.execute(f"""
        INSERT OR REPLACE INTO {individual_details_table} 
        (ID, location, camera_ids, num_frames, img_names)
        VALUES (?, ?, ?, ?, ?);
        """, (individual_id, location, camera_ids, num_frames, img_names))

    # Commit all changes to the database
    conn.commit()


def postprocess_database(db_path, source_table):
    """
    This function processes the SQLite database, creating three tables: individual summary by location,
    camera summary by location and camera_id, and individual details for each detected ID.

    Args:
    db_path (str): Path to the SQLite database.
    source_table (str): Name of the table that holds the detection data.

    Returns:
    None
    """
    conn = sqlite3.connect(db_path)

    try:
        # Create the summary and individual details tables
        create_individual_summary_table(conn, 'location_summary')
        create_camera_summary_table(conn, 'camera_summary')
        create_individual_details_table(conn, 'individual_details')

        # Process and insert data into these tables
        process_individuals_data(conn, source_table, 'location_summary', 'camera_summary', 'individual_details')

        print("Database processing complete.")
    except Exception as e:
        print(f"Error processing the database: {e}")
    finally:
        conn.close()


def postprocess_sqlite_data(db_path: str, source_table: str):
    """
    Post-process the SQLite database to create summary tables and individual details.
    Callback to be invoked at the processing end of the pipeline.

    Args:
        db_path (str): Path to the SQLite database.
        source_table (str): Name of the table that holds the detection data.

    Returns:
        Callable: A callback function that processes the database.
    """

    def _callback(predictor: Optional[Callable] = None):
        """
        Callback function to be called after the detection pipeline.
        """
        try:
            postprocess_database(db_path, source_table)
        except Exception as e:
            print(f"[sqlite_db_callback] Error processing data: {e}")

    return _callback