"""MobileNetV3 ImageNet classification baseline used before detection."""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def predict(image_path: Path, topk: int = 5):
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    tensor = weights.transforms()(image).unsqueeze(0)
    with torch.no_grad():
        probabilities = model(tensor).softmax(dim=1)[0]

    scores, labels = torch.topk(probabilities, topk)
    categories = weights.meta["categories"]
    return [(categories[int(label)], float(score)) for label, score in zip(labels, scores)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="Image file paths.")
    parser.add_argument("--topk", default=5, type=int)
    return parser.parse_args()


def main(args):
    for image in args.images:
        print(image)
        for label, score in predict(Path(image), topk=args.topk):
            print(f"  {label}: {score:.4f}")


if __name__ == "__main__":
    main(parse_args())
