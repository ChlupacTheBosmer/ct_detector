# ct_detector/data/dataset.py

import os
import logging
import random
import shutil
import traceback
import re
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

from ultralytics.data.utils import img2label_paths
from ultralytics.utils.checks import check_yaml
from ultralytics.utils import yaml_load
from ultralytics.engine.results import Boxes

from ct_detector.utils.files import load_image
from ct_detector.data.utils import get_image_paths_recursive, filter_paths_by_name, load_labels_for_images


class Dataset(dict):
    """
    Lightweight dictionary-like class for managing YOLO-compatible datasets using Ultralytics tools.
    Automatically pairs images with YOLO-style label files, loading annotations into Boxes.
    """

    def __init__(self, *args, **kwargs):
        """
        Allow initialization from a dictionary like the built-in dict().
        """
        super().__init__(*args, **kwargs)
        self.dataset_name = None

    @classmethod
    def from_folder(cls, folder_path: Union[str, Path], valid_exts={".jpg", ".jpeg", ".png"},
                    exclude_names_path: Optional[Union[str, Path]] = None,
                    threads: int = 8) -> "Dataset":
        dataset = cls()
        folder_path = Path(folder_path)
        dataset.dataset_name = folder_path.name
        all_images = get_image_paths_recursive(folder_path, valid_exts)
        if exclude_names_path:
            all_images = filter_paths_by_name(all_images, exclude_names_path)
        with ThreadPoolExecutor(max_workers=threads) as executor:
            results = list(executor.map(load_labels_for_images, all_images))
        for meta in results:
            if meta:
                dataset[meta["name"]] = meta
        return dataset

    @classmethod
    def from_txt(cls, txt_file: Union[str, Path],
                 exclude_names_path: Optional[Union[str, Path]] = None) -> "Dataset":
        """
        Load dataset from a YOLO-style .txt file (one image path per line).
        If paths are relative, they are resolved relative to the .txt file location.
        """
        txt_file = Path(txt_file)
        base_dir = txt_file.parent
        dataset = cls()

        with txt_file.open("r") as f:
            image_paths = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path = Path(line)
                if not path.is_absolute():
                    path = base_dir / path  # resolve relative to txt file
                image_paths.append(path)

        if exclude_names_path:
            image_paths = filter_paths_by_name(image_paths, exclude_names_path)

        for img_path in image_paths:
            meta = load_labels_for_images(img_path)
            if meta:
                dataset[meta["name"]] = meta

        return dataset

    @classmethod
    def from_yaml(cls, yaml_file: Union[str, Path]) -> "Dataset":
        """
        Load a YOLO-format dataset from a .yaml config file.
        Resolves relative paths properly based on the .yaml file location.
        """
        yaml_file = Path(yaml_file)
        data_cfg = yaml_load(check_yaml(yaml_file))
        dataset = cls()
        dataset.dataset_name = yaml_file.stem

        # Resolve dataset root
        yaml_dir = yaml_file.parent.resolve()
        dataset_root = (yaml_dir / data_cfg.get("path")).resolve() if data_cfg.get("path") else yaml_dir

        for split in ["train", "val", "test"]:
            split_value = data_cfg.get(split)
            if not split_value:
                continue  # skip missing/null entries

            split_path = Path(split_value)
            if not split_path.is_absolute():
                split_path = (dataset_root / split_path).resolve()

            sub_dataset = cls.from_txt(split_path)
            for k, v in sub_dataset.items():
                v["dataset"] = split
                dataset[k] = v

        return dataset

    @classmethod
    def from_paths(cls, image_paths: List[Union[str, Path]]) -> "Dataset":
        dataset = cls()
        for img_path in image_paths:
            meta = load_labels_for_images(Path(img_path))
            if meta:
                dataset[meta["name"]] = meta
        return dataset

    @classmethod
    def from_datasets(cls, datasets: List["Dataset"]) -> "Dataset":
        merged = cls()
        for dset in datasets:
            for k, v in dset.items():
                merged[k] = v
        merged.dataset_name = "_".join([d.dataset_name for d in datasets if d.dataset_name])
        return merged

    def filter_by(self, key: str, exclude_values: List[str]) -> None:
        to_remove = [k for k, v in self.items() if v.get(key) in exclude_values]
        for k in to_remove:
            self.pop(k)

    def get_subset(self, subset_type: str) -> "Dataset":
        subset = Dataset()
        for key, value in self.items():
            if value.get("dataset") == subset_type:
                subset[key] = value
        return subset

    def get_random_subset(self, size: int, balanced: bool = False, by: str = 'dataset') -> "Dataset":
        if balanced:
            grouped = defaultdict(list)
            for k, v in self.items():
                grouped[v.get(by)].append(k)
            result = Dataset()
            while len(result) < size:
                for group in grouped.values():
                    if group:
                        choice = random.choice(group)
                        result[choice] = self[choice]
                        group.remove(choice)
                        if len(result) == size:
                            break
            return result
        else:
            chosen = random.sample(list(self.keys()), size)
            return Dataset({k: self[k] for k in chosen})

    def split_dataset(self, train_ratio: float, val_ratio: float, test_ratio: float = 0.0):
        all_keys = list(self.keys())
        random.shuffle(all_keys)
        n = len(all_keys)
        t1 = int(train_ratio * n)
        t2 = t1 + int(val_ratio * n)
        for i, k in enumerate(all_keys):
            self[k]['dataset'] = 'train' if i < t1 else 'val' if i < t2 else 'test'

    def class_distribution(self, class_names: Dict[int, str]) -> Dict[str, int]:
        dist = {name: 0 for name in class_names.values()}
        for v in self.values():
            if 'labels' in v:
                for cls in v['labels']:
                    dist[class_names[int(cls)]] += 1
        return dist

    def reorganize_files(self, base_folder: Union[str, Path], by: str):
        for v in self.values():
            new_dir = Path(base_folder) / str(v.get(by, 'unknown'))
            new_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(v['path'], new_dir / v['name'])
            if 'label_path' in v and v['label_path']:
                shutil.copy(v['label_path'], new_dir / Path(v['label_path']).name)

    def generate_yolo_files(self, out_dir: Union[str, Path], classes: Dict[int, str], abs_paths=False, write_yaml=True):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {k: [] for k in ['train', 'val', 'test']}

        for v in self.values():
            if v.get('dataset') in paths:
                img_path = Path(v['path']).resolve()
                if abs_paths:
                    paths[v['dataset']].append(str(img_path))
                else:
                    # Safely compute relative path from the out_dir to image path
                    try:
                        rel_path = os.path.relpath(img_path, start=out_dir.resolve())
                    except ValueError:
                        # fallback in case drives mismatch on Windows
                        rel_path = str(img_path)
                    paths[v['dataset']].append(rel_path)

        for k, lines in paths.items():
            with open(out_dir / f"{k}.txt", 'w') as f:
                f.write("\n".join(lines))

        if write_yaml:
            yaml_path = out_dir / f"{self.dataset_name or 'dataset'}.yaml"
            with open(yaml_path, 'w') as f:
                yaml_content = {
                    'path': str(out_dir.resolve()),
                    'train': 'train.txt',
                    'val': 'val.txt',
                    'test': 'test.txt' if paths['test'] else None,
                    'names': classes
                }
                # remove 'test' if empty
                yaml_content = {k: v for k, v in yaml_content.items() if v is not None}
                import yaml
                yaml.dump(yaml_content, f)

    def visualize(self, key: str, color_conversion: Optional[str] = None):
        try:
            from ultralytics.engine.results import Results
            import matplotlib.pyplot as plt

            v = self[key]
            img = load_image(v['path'], conversion=color_conversion)[0]
            r = Results(img, v['path'], {i: str(i) for i in range(101)}, boxes=v.get('boxes').data if v.get('boxes') else None)
            plt.imshow(r.plot())
            plt.axis('off')
            plt.show()
        except Exception as e:
            logging.error(f"Visualization failed for {key}: {e}")
            traceback.print_exc()

    def sanity_check(self):
        missing = []
        corrupted = []
        for v in self.values():
            if v.get('labels') and not os.path.exists(v.get('label_path', '')):
                missing.append(v['name'])
            try:
                load_image(v['path'])
            except Exception:
                corrupted.append(v['name'])
        return missing, corrupted

    def balance_by_class(self, target_size: int) -> "Dataset":
        by_class = defaultdict(list)
        for k, v in self.items():
            for cls in v.get('labels', []):
                by_class[int(cls)].append(k)
        selected = []
        for v in by_class.values():
            selected.extend(random.sample(v, min(target_size, len(v))))
        return Dataset({k: self[k] for k in selected})

    def export_annotations(self, fmt: str, out_dir: Union[str, Path]):
        raise NotImplementedError("Export to other formats is not implemented.")

    @property
    def size(self):
        return len(self)

    @property
    def train_size(self):
        return sum(1 for v in self.values() if v.get('dataset') == 'train')

    @property
    def val_size(self):
        return sum(1 for v in self.values() if v.get('dataset') == 'val')

    @property
    def test_size(self):
        return sum(1 for v in self.values() if v.get('dataset') == 'test')

    @property
    def with_detection(self):
        return sum(1 for v in self.values() if v.get("boxes") is not None)

    @property
    def without_detection(self):
        return self.size - self.with_detection

    @property
    def dataset_names(self):
        return set(v['dataset'] for v in self.values() if 'dataset' in v)

    @property
    def parent_folders(self):
        return set(Path(v['path']).parent.name for v in self.values())
