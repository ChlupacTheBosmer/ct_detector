from pathlib import Path
import cv2
import re
from ultralytics.models.yolo.detect.predict import DetectionPredictor

class CtPredictor(DetectionPredictor):
    """
    A custom Ultralytics predictor subclass that:
    1) Allows skipping, renaming, or overwriting existing label (.txt) files.
    2) Optionally directs label files to a custom directory (labels_dir).

    This version is updated to match the newer BasePredictor write_results(...)
    signature:
        write_results(self, i, p, im, s)
    """

    def __init__(
        self,
        overrides=None,
        _callbacks=None,
        handle_existing_labels: str = "overwrite",
        labels_dir: str = ""
    ):
        """
        Args:
            overrides: Any arguments you want to pass to the predictor (e.g. conf=0.25).
            _callbacks: The Ultralytics callback system. If None, a default is created.
            handle_existing_labels (str):
                'skip':    if label file exists, do not create/overwrite new label
                'rename':  if label file exists, rename it to .txt.old
                'overwrite': overwrite if label file exists (default)
            labels_dir (str): Custom folder to store .txt label files (instead
                              of the default runs/predict/exp/labels).
        """
        super().__init__(overrides=overrides, _callbacks=_callbacks)
        assert handle_existing_labels in ["skip", "rename", "overwrite"], \
            f"Invalid handle_existing_labels: {handle_existing_labels}. Must be one of: skip, rename, overwrite."
        self.handle_existing_labels = handle_existing_labels
        self.labels_dir = labels_dir

    def write_results(self, i, p, im, s):
        """
        Override of the new base 'write_results(i, p, im, s)' method to handle:
         - skip/rename/overwrite existing label files
         - custom labels_dir if desired

        Args:
            i (int): Index of the current image in the batch.
            p (Path): Path to the current image (e.g. 'images/img001.jpg').
            im (torch.Tensor): The entire preprocessed batch (shape = [B, C, H, W]).
            s (List[str]): List of logging strings for each image in the batch.

        Returns:
            (str): A logging string with result info, same as base method.
        """
        # First, let's see how the base class would parse the 'frame' index:
        # In the new code, for streams/from_img/tensor, it sets frame = self.dataset.count.
        # Otherwise, it tries to parse e.g. "frame 10/300" from s[i].
        # We'll replicate that logic:

        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:
            frame = self.dataset.count
        else:
            match = re.search(r"frame (\d+)/", s[i])
            frame = int(match[1]) if match else None  # or None if not found

        # The base code sets self.txt_path to something like:
        #   self.save_dir / 'labels' / (p.stem + f'_{frame}' if video)
        # We'll build our own label_file path. Then we'll do skip/rename checks, then let the
        # base method do its normal routine.

        # Start with either base predictor's label folder or our custom folder
        if self.labels_dir:
            label_folder = Path(self.labels_dir)
        else:
            label_folder = self.save_dir / 'labels'
        label_folder.mkdir(parents=True, exist_ok=True)

        label_file = label_folder / p.stem
        if self.dataset and self.dataset.mode != 'image' and frame is not None:
            # For video/stream, add _frame to the filename
            label_file = label_file.with_name(label_file.stem + f"_{frame}" + '.txt')
        else:
            label_file = Path(str(label_file) + '.txt')

        # Check if it exists
        if label_file.is_file():
            if self.handle_existing_labels == "skip":
                skip_label = True
            elif self.handle_existing_labels == "rename":
                new_name = label_file.with_suffix(".txt.old")
                label_file.rename(new_name)
                skip_label = False
            else:
                skip_label = False
        else:
            skip_label = False

        # Temporarily override self.txt_path so the base class writes to our new location
        original_txt_path = self.txt_path
        self.txt_path = str(label_file.with_suffix(''))  # The base method appends .txt internally

        # If skipping, temporarily disable label writing for this image
        original_save_txt = self.args.save_txt
        if skip_label:
            self.args.save_txt = False

        # Now we can call the base method to do all normal logic (plotting, saving .txt/crops, etc.)
        string = ""  # print string
        try:
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
                string += f"{i}: "
                frame = self.dataset.count
            else:
                match = re.search(r"frame (\d+)/", s[i])
                frame = int(match[1]) if match else None  # 0 if frame undetermined

            #self.txt_path = self.save_dir / "labels" / (p.stem + ("" if self.dataset.mode == "image" else f"_{frame}"))
            string += "{:g}x{:g} ".format(*im.shape[2:])
            result = self.results[i]
            result.save_dir = self.save_dir.__str__()  # used in other locations
            string += f"{result.verbose()}{result.speed['inference']:.1f}ms"

            # Add predictions to image
            if self.args.save or self.args.show:
                self.plotted_img = result.plot(
                    line_width=self.args.line_width,
                    boxes=self.args.show_boxes,
                    conf=self.args.show_conf,
                    labels=self.args.show_labels,
                    im_gpu=None if self.args.retina_masks else im[i],
                )

            # Save results
            if self.args.save_txt:
                result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
            if self.args.save_crop:
                result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)
            if self.args.show:
                self.show(str(p))
            if self.args.save:
                self.save_predicted_images(str(self.save_dir / p.name), frame)
        finally:
            # Restore the original state
            self.txt_path = original_txt_path
            self.args.save_txt = original_save_txt

        return string

