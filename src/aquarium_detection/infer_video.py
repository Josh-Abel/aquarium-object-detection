"""Run Faster R-CNN aquarium detections on a video file."""

import argparse
from pathlib import Path

import cv2
import torch
import torchvision.transforms.v2 as T

from .config import AQUARIUM_CLASSES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES
from .faster_rcnn import fasterrcnn_mobilenet_v3


def build_frame_transform():
    return T.Compose(
        [
            T.ToImage(),
            T.Resize(size=(IMAGE_SIZE, IMAGE_SIZE), antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            T.ToPureTensor(),
        ]
    )


def load_model(checkpoint_path: str, device: torch.device, num_classes: int):
    model = fasterrcnn_mobilenet_v3(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def draw_predictions(frame, detections, score_threshold: float):
    height, width = frame.shape[:2]
    x_scale = width / IMAGE_SIZE
    y_scale = height / IMAGE_SIZE

    for box, label, score in zip(detections["boxes"], detections["labels"], detections["scores"]):
        if float(score) < score_threshold:
            continue

        x1, y1, x2, y2 = box.detach().cpu().tolist()
        x1, x2 = int(x1 * x_scale), int(x2 * x_scale)
        y1, y2 = int(y1 * y_scale), int(y2 * y_scale)
        class_name = AQUARIUM_CLASSES.get(int(label), f"class_{int(label)}")
        text = f"{class_name} {float(score):.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(frame, text, (x1, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)

    return frame


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument("--output", default="outputs/video_predictions.mp4")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--score-threshold", default=0.5, type=float)
    parser.add_argument("--num-classes", default=NUM_CLASSES, type=int)
    return parser.parse_args()


@torch.inference_mode()
def main(args):
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device=device, num_classes=args.num_classes)
    transform = build_frame_transform()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {args.input}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = transform(rgb_frame).to(device)
        detections, _ = model([image])
        annotated = draw_predictions(frame, detections[0], args.score_threshold)
        writer.write(annotated)

    cap.release()
    writer.release()
    print(f"Processed {frame_count} frames -> {output_path}")


if __name__ == "__main__":
    main(parse_args())
