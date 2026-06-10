"""Earlier single-shark MobileNetV3 classifier/regressor prototype."""

import torch
from torch import nn
from torchvision import models


class SingleSharkDetector(nn.Module):
    """Predict one normalized AABB and a binary shark/no-shark score."""

    def __init__(self, num_classes=1, regressor_dropout=0.3, classifier_dropout=0.5):
        super().__init__()
        self.mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        in_features = self.mobilenet.classifier[0].in_features

        for param in self.mobilenet.parameters():
            param.requires_grad = False

        self.regressor = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(regressor_dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(regressor_dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(regressor_dropout),
            nn.Linear(64, 4),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(64, num_classes),
            nn.Sigmoid(),
        )
        self.mobilenet.classifier = nn.Identity()

    def forward(self, x):
        features = self.mobilenet(x)
        return self.regressor(features), self.classifier(features)


def normalize_bbox(bbox, image_size):
    x_min, y_min, x_max, y_max = bbox
    width, height = image_size
    return [x_min / width, y_min / height, x_max / width, y_max / height]


def unnormalize_bbox(bbox, image_size):
    x_min, y_min, x_max, y_max = bbox
    width, height = image_size
    return [int(x_min * width), int(y_min * height), int(x_max * width), int(y_max * height)]


def compute_iou(box_a, box_b, eps=1e-6):
    x_a = torch.max(box_a[0], box_b[0])
    y_a = torch.max(box_a[1], box_b[1])
    x_b = torch.min(box_a[2], box_b[2])
    y_b = torch.min(box_a[3], box_b[3])

    inter_area = torch.clamp(x_b - x_a, min=0) * torch.clamp(y_b - y_a, min=0)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter_area / (box_a_area + box_b_area - inter_area + eps)
