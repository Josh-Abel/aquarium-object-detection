"""Convert a Roboflow TensorFlow CSV object-detection export to COCO JSON."""

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

from .config import AQUARIUM_CLASSES


OBJECT_CLASS_TO_ID = {name: class_id for class_id, name in AQUARIUM_CLASSES.items() if class_id != 0}


def clamp(value, low, high):
    return max(low, min(high, value))


def convert_split(split_dir: Path, output_name: str) -> tuple[int, int]:
    csv_path = split_dir / "_annotations.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {csv_path}")

    images = OrderedDict()
    annotations = []
    annotation_id = 1

    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing_columns)}")

        for row in reader:
            filename = row["filename"].strip()
            class_name = row["class"].strip()
            if class_name not in OBJECT_CLASS_TO_ID:
                raise ValueError(f"Unknown class {class_name!r} in {csv_path}")

            width = int(float(row["width"]))
            height = int(float(row["height"]))
            xmin = clamp(float(row["xmin"]), 0, width)
            ymin = clamp(float(row["ymin"]), 0, height)
            xmax = clamp(float(row["xmax"]), 0, width)
            ymax = clamp(float(row["ymax"]), 0, height)
            box_width = xmax - xmin
            box_height = ymax - ymin
            if box_width <= 0 or box_height <= 0:
                continue

            if filename not in images:
                image_id = len(images) + 1
                images[filename] = {
                    "id": image_id,
                    "file_name": filename,
                    "width": width,
                    "height": height,
                }

            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": images[filename]["id"],
                    "category_id": OBJECT_CLASS_TO_ID[class_name],
                    "bbox": [xmin, ymin, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    coco = {
        "images": list(images.values()),
        "annotations": annotations,
        "categories": [
            {"id": class_id, "name": class_name, "supercategory": "aquarium"}
            for class_id, class_name in AQUARIUM_CLASSES.items()
            if class_id != 0
        ],
    }

    output_path = split_dir / output_name
    with output_path.open("w") as output_file:
        json.dump(coco, output_file)

    return len(images), len(annotations)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        required=True,
        help="Dataset root containing train/valid/test split folders with _annotations.csv files.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--output-name", default="_annotations.coco.json")
    return parser.parse_args()


def main(args):
    data_root = Path(args.data_root)
    for split in args.splits:
        split_dir = data_root / split
        image_count, annotation_count = convert_split(split_dir, args.output_name)
        print(f"{split}: wrote {image_count} images and {annotation_count} annotations")


if __name__ == "__main__":
    main(parse_args())
