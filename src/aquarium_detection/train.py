"""Train or evaluate the MobileNetV3 Faster R-CNN aquarium detector."""

import argparse
import datetime
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from .augmentations import train_augmentations, val_augmentations
from .coco_utils import get_coco
from .config import NUM_CLASSES
from .engine import evaluate, train_one_epoch, visualize_detections
from .faster_rcnn import fasterrcnn_mobilenet_v3
from .group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from . import utils


def get_summary_writer(output_dir: str) -> SummaryWriter:
    log_dir = os.path.join(output_dir, "tensorboard_logs", str(int(time.time())))
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir)


def get_dataset(is_train: bool, args: argparse.Namespace):
    image_set = "train" if is_train else "val"
    transforms = train_augmentations() if is_train else val_augmentations()
    dataset = get_coco(root=args.data_path, image_set=image_set, transforms=transforms)
    return dataset, args.num_classes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True, help="COCO-format dataset root containing train/ and valid/.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--epochs", default=26, type=int)
    parser.add_argument("--lr", default=0.05, type=float)
    parser.add_argument("--weight-decay", default=1e-5, type=float, dest="weight_decay")
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--num-classes", default=NUM_CLASSES, type=int)
    parser.add_argument("--print-freq", default=20, type=int)
    parser.add_argument("--output-dir", default="outputs", type=str)
    parser.add_argument("--resume", default="", help="Checkpoint path for resume/evaluation.")
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--visualize", action="store_true", help="Save sample prediction images during test-only runs.")
    parser.add_argument("--conf-threshold", default=0.2, type=float)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if args.output_dir:
        utils.mkdir(args.output_dir)

    device = torch.device(args.device)
    print(f"Using device: {device}")
    print("Loading data")

    dataset, num_classes = get_dataset(is_train=True, args=args)
    dataset_test, _ = get_dataset(is_train=False, args=args)

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    group_ids = create_aspect_ratio_groups(dataset, k=3)
    train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=train_batch_sampler,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn,
    )

    print("Creating model")
    model = fasterrcnn_mobilenet_v3(num_classes=num_classes)
    model.to(device)

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 25], gamma=0.1)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1

    if args.test_only:
        evaluate(model, data_loader_test, device=device)
        if args.visualize:
            visualize_detections(
                model=model,
                data_loader=data_loader_test,
                device=device,
                output_dir=os.path.join(args.output_dir, "predictions"),
                num_images=10,
                conf_threshold=args.conf_threshold,
            )
        return

    print("Start training")
    writer = get_summary_writer(args.output_dir)
    start_time = time.time()
    best_map = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq, writer=writer)
        coco_evaluator, _ = evaluate(model, data_loader_test, device=device, epoch=epoch, writer=writer)
        lr_scheduler.step()

        map_metric = coco_evaluator.coco_eval["bbox"].stats[0]
        print(f"Epoch {epoch} | mAP: {map_metric:.4f}")

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "args": args,
            "epoch": epoch,
        }
        latest_path = os.path.join(args.output_dir, "checkpoint_latest.pth")
        utils.save_on_master(checkpoint, latest_path)

        if map_metric > best_map:
            best_map = map_metric
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "best_model.pth"))
            print(f"New best model saved with mAP: {best_map:.4f}")

    writer.close()
    total_time = time.time() - start_time
    print(f"Training time {datetime.timedelta(seconds=int(total_time))}")


if __name__ == "__main__":
    main(parse_args())
