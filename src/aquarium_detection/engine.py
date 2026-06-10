import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
from . import utils
from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset
from .config import IMAGENET_MEAN, IMAGENET_STD

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None, writer=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
    # Initialize tracking for tensorboard
    tb_loss_dict = {}
    tb_total_loss = 0.0
    tb_steps = 0

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        tb_steps += 1
        images = list(image.to(device) for image in images)

        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # Accumulate loss values for TensorBoard
        tb_total_loss += losses_reduced.item()
        for k, v in loss_dict_reduced.items():
            if k not in tb_loss_dict:
                tb_loss_dict[k] = 0.0
            tb_loss_dict[k] += v.item()

        # Log to TensorBoard using metric_logger values
        if writer is not None:
            # Log learning rate
            writer.add_scalar('LR_btach', optimizer.param_groups[0]["lr"], tb_steps)
            
            # Log total loss (from losses_reduced)
            writer.add_scalar('Loss/train_btach', losses_reduced.item(), tb_steps)
            
            # Log each component (from loss_dict_reduced)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}_btach', v.item(), tb_steps)
    
    # Log to TensorBoard using metric_logger values
    if writer is not None:
        # Log learning rate
        writer.add_scalar('LR', optimizer.param_groups[0]["lr"], tb_steps)
        
        # Log total loss (from losses_reduced)
        writer.add_scalar('Loss/train_epoch', tb_total_loss / tb_steps, epoch)
        
        # Log each component (from loss_dict_reduced)
        for k, v in tb_loss_dict.items():
            writer.add_scalar(f'Loss/{k}_epoch', v / tb_steps, epoch)
            
    return metric_logger




@torch.inference_mode()
def evaluate(model, data_loader, device, epoch=None, writer=None):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)

    coco_evaluator = CocoEvaluator(coco, ["bbox"])
    # Initialize tracking for tensorboard
    tb_val_loss_dict = {}
    tb_val_total_loss = 0.0
    tb_val_steps = 0

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        tb_val_steps += 1
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs, loss_dict = model(images, targets)
        
        losses = sum(loss for loss in loss_dict.values())
            
        # Reduce losses over all GPUs
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        
        # Accumulate validation loss values for TensorBoard
        tb_val_total_loss += losses_reduced
        for k, v in loss_dict_reduced.items():
            if k not in tb_val_loss_dict:
                tb_val_loss_dict[k] = 0.0
            tb_val_loss_dict[k] += v.item()
        
        
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        
        # Log validation metrics to TensorBoard
        if writer is not None and tb_val_steps > 0:
        # Log total validation loss
            writer.add_scalar('Loss/val_batch', losses_reduced.item(), tb_val_steps)
            
            # Log each validation loss component
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss_val/{k}_batch', v.item() , tb_val_steps)
        
        
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    
    if writer is not None:
        writer.add_scalar('Loss/val_epoch', tb_val_total_loss / tb_val_steps, epoch)
            
        # Log each validation loss component
        for k, v in tb_val_loss_dict.items():
            writer.add_scalar(f'Loss_val/{k}_epoch', v / tb_val_steps, epoch)
        # Log all AP metrics (Precision)
        writer  .add_scalar('COCO/mAP_all', coco_evaluator.coco_eval['bbox'].stats[0], epoch)
        writer.add_scalar('COCO/mAP_50', coco_evaluator.coco_eval['bbox'].stats[1], epoch)
        writer.add_scalar('COCO/mAP_75', coco_evaluator.coco_eval['bbox'].stats[2], epoch)
        writer.add_scalar('COCO/mAP_small', coco_evaluator.coco_eval['bbox'].stats[3], epoch)
        writer.add_scalar('COCO/mAP_medium', coco_evaluator.coco_eval['bbox'].stats[4], epoch)
        writer.add_scalar('COCO/mAP_large', coco_evaluator.coco_eval['bbox'].stats[5], epoch)
        
        # Log all AR metrics (Recall)
        writer.add_scalar('COCO/mAR_1', coco_evaluator.coco_eval['bbox'].stats[6], epoch)
        writer.add_scalar('COCO/mAR_10', coco_evaluator.coco_eval['bbox'].stats[7], epoch)
        writer.add_scalar('COCO/mAR_100', coco_evaluator.coco_eval['bbox'].stats[8], epoch)
        writer.add_scalar('COCO/mAR_small', coco_evaluator.coco_eval['bbox'].stats[9], epoch)
        writer.add_scalar('COCO/mAR_medium', coco_evaluator.coco_eval['bbox'].stats[10], epoch)
        writer.add_scalar('COCO/mAR_large', coco_evaluator.coco_eval['bbox'].stats[11], epoch)

    return coco_evaluator, tb_val_total_loss / tb_val_steps

def visualize_detections(model, data_loader, device, output_dir=None, num_images=10, conf_threshold=0.5):
    """
    Visualize model detections against ground truth annotations.
    
    Args:
        model: The detection model
        data_loader: Data loader with images and annotations
        device: Device to run inference on
        output_dir: Directory to save visualizations (if None, will display)
        num_images: Maximum number of images to visualize
        conf_threshold: Confidence threshold for showing predictions
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    import os
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model.eval()
    cpu_device = torch.device("cpu")
    
    visualized_count = 0
    
    for images, targets in data_loader:
        if visualized_count >= num_images:
            break
            
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        # Keep original images for visualization
        orig_images = [img.clone().to(cpu_device) for img in images]
        
        # Get model predictions
        with torch.no_grad():
            outputs, _ = model(images, targets)
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        
        # Visualize each image
        for i, (img, target, output) in enumerate(zip(orig_images, targets, outputs)):
            if visualized_count >= num_images:
                break
                
            # Convert tensor to numpy image
            img_np = img.permute(1, 2, 0).cpu().numpy()
            mean = np.array(IMAGENET_MEAN).reshape(1, 1, 3)
            std = np.array(IMAGENET_STD).reshape(1, 1, 3)
            img_np = np.clip(img_np * std + mean, 0, 1)

            # Create figure
            fig, ax = plt.subplots(1, figsize=(12, 9))
            ax.imshow(img_np)
            
            # Draw ground truth boxes
            for box, label in zip(target['boxes'], target['labels']):
                x1, y1, x2, y2 = box.cpu().numpy()
                # Adjust the category lookup based on your dataset
                try:
                    category = data_loader.dataset.coco.cats[label.item()]['name']
                except:
                    category = f"Class {label.item()}"
                    
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='g', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1, f"GT: {category}", fontsize=9, color='white', 
                        bbox=dict(facecolor='green', alpha=0.5))
            
            # Draw predictions
            for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
                if score > conf_threshold:
                    x1, y1, x2, y2 = box.cpu().numpy()
                    try:
                        category = data_loader.dataset.coco.cats[label.item()]['name']
                    except:
                        category = f"Class {label.item()}"
                        
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x1, y1-10, f"Pred: {category} ({score:.2f})", fontsize=9, color='white',
                            bbox=dict(facecolor='red', alpha=0.5))
            
            image_id = target.get('image_id', visualized_count)
            
            # Save or display
            if output_dir:
                plt.savefig(os.path.join(output_dir, f"detection_{image_id}.png"))
                plt.close()
            else:
                plt.tight_layout()
                plt.show()
            
            visualized_count += 1
    
    print(f"Visualized {visualized_count} images")


def visualize_detections_on_video(model, video_path, device, transform=None, output_path=None, conf_threshold=0.5, display=True):
    """
    Process a video file to run model detections on each frame and visualize predictions.
    
    Args:
        model: The detection model.
        video_path: Path to the input video file.
        device: Device to run inference on.
        output_path: If provided, path to save the annotated video.
        conf_threshold: Confidence threshold to filter predictions.
        display: If True, display each frame in a window.
    """
    import cv2
    import torch
    import torchvision.transforms as transforms
    import numpy as np
    try:
        from sahi.predict import get_sliced_prediction
    except ImportError:
        from sahi.sahi.predict import get_sliced_prediction

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    colors = [
        (0, 0, 255),   # Red
        (0, 255, 0),   # Green
        (255, 0, 0),   # Blue
        (0, 255, 255), # Yellow
        (255, 0, 255), # Magenta
        (255, 255, 0), # Cyan
        (128, 0, 128),  # Purple
        (128, 128, 128)  # Purple
    ]
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set up video writer if output_path is provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        frame_count += 1

        # Convert frame (BGR) to RGB and then to tensor
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Write frame to output video if desired
        
        result = get_sliced_prediction(
                                        frame_rgb,
                                        model,
                                        slice_height = 1280,
                                        slice_width = 1280,
                                        overlap_height_ratio = 0.1,
                                        overlap_width_ratio = 0.1
                                    )
        result_dict = result.export_visuals()
        frame = np.array(result_dict["image"])
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


        if output_path:
            out.write(frame)

        # Display frame if requested
        if display:
            cv2.imshow('Detections', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    # Define a transform to convert frames to tensor
    transform = transforms.ToTensor()

    cpu_device = torch.device("cpu")

    while not True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        frame_count += 1

        # Convert frame (BGR) to RGB and then to tensor
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = transform(frame_rgb).to(device)
        id2name = {'0':'None','1':'fish','2':'jellyfish', '3':'penguin', '4':'puffin', '5':'shark', '6': 'starfish', '7':'stingray'}
        # The model typically expects a list of tensors
        with torch.no_grad():
            outputs = model([img_tensor])[0]
            for output in outputs:
                # Ensure outputs are on CPU for drawing
                boxes = output['boxes'].to(cpu_device)
                labels = output['labels'].to(cpu_device)
                scores = output['scores'].to(cpu_device)

                # Draw prediction boxes on the original frame (still in BGR)
                for box, label, score in zip(boxes, labels, scores):
                    if score > conf_threshold:
                        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                        # You can modify this if you have a mapping for label -> class name
                        category = f"Class {id2name[f'{label.item()}']}"
                        # Draw rectangle (red color)
                        color = colors[label.item()]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        # Put text above the box
                        text = f"Pred: {category} ({score:.2f})"

                        cv2.putText(frame, text, (x1, max(y1-10, 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, color, 2)

            # Write frame to output video if desired
            if output_path:
                out.write(frame)

        # Display frame if requested
        if display:
            cv2.imshow('Detections', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames.")
