"""Project-level constants for the aquarium detector."""

AQUARIUM_CLASSES = {
    0: "background",
    1: "fish",
    2: "jellyfish",
    3: "penguin",
    4: "puffin",
    5: "shark",
    6: "starfish",
    7: "stingray",
}

NUM_CLASSES = len(AQUARIUM_CLASSES)
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
