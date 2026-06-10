# Aquarium Object Detection

Multi-class aquarium object detection pipeline built with PyTorch. The final detector uses a MobileNetV3 backbone inside a Faster R-CNN style model for detecting aquarium animals with axis-aligned bounding boxes.

This repository keeps the implemented detection pipeline focused and reproducible: dataset loading in COCO format, data augmentation, MobileNetV3/FPN feature extraction, RPN and ROI heads, NMS-based detection, TensorBoard logging, COCO mAP evaluation, sample visualizations, and video inference. Large datasets, model weights, videos, and cache files are intentionally excluded.

## Dataset

Where to download the data from: https://public.roboflow.com/object-detection/aquarium

The local dataset is the Roboflow Aquarium object-detection export in TensorFlow CSV format:

```text
data/Aquarium Combined.v2-raw-1024.tensorflow/
  train/_annotations.csv
  valid/_annotations.csv
  test/_annotations.csv
```

Convert it once to COCO JSON before training:

```bash
python -m aquarium_detection.convert_roboflow_csv_to_coco \
  --data-root "data/Aquarium Combined.v2-raw-1024.tensorflow"
```

After conversion, the training code reads the same dataset folder with COCO annotation files:

```text
data/Aquarium Combined.v2-raw-1024.tensorflow/
  train/
    _annotations.coco.json
    *.jpg
  valid/
    _annotations.coco.json
    *.jpg
```

Classes used by the final detector:

```text
background, fish, jellyfish, penguin, puffin, shark, starfish, stingray
```

The repository also preserves a small single-shark prototype trained from CSV annotations for context, but the final pipeline focuses on multi-class aquarium detection.

## Model Architecture

- Backbone: pretrained MobileNetV3 Large with ImageNet weights.
- Feature extraction: MobileNetV3 layers wrapped with an FPN.
- Detector: Faster R-CNN style model with a Region Proposal Network and ROI heads.
- Boxes: axis-aligned bounding boxes in `xyxy` format.
- Classification/regression heads: cross-entropy classification and Smooth L1 box regression losses.
- Inference filtering: confidence thresholding and NMS through the ROI head implementation.

## Training Approach

Training uses PyTorch and torchvision transforms:

- Resize to `224 x 224`.
- Random horizontal/vertical flips, random affine rotation, channel/photometric transforms, blur, and grayscale variants.
- AdamW optimizer.
- Multi-step learning rate schedule.
- TensorBoard logging for training loss, validation loss, and COCO metrics.
- Checkpointing for the latest model and best validation mAP.

## Evaluation Metrics

The evaluation path uses COCO-style bounding-box metrics through `pycocotools`, including:

- mAP at IoU `0.50:0.95`
- mAP at IoU `0.50`
- mAP at IoU `0.75`
- Average recall metrics

The cleaned training/evaluation code can regenerate these metrics when a validated checkpoint is available. The archived results below document working runs from the original implementation; the exact training logs and weights are not included.

## Results

Demo outputs:

- Multi Class Multi Object Detection: https://youtube.com/shorts/rv-Fisr1evg?feature=share
- Single Class Multi Object Detection: https://youtube.com/shorts/YNGHkSCyJsk?feature=share
- Single Class Single Object Detection: https://youtu.be/938D9_5eWoI

Archived evaluation results:

| Model variant | mAP `0.50:0.95` | mAP `0.50` | mAP `0.75` | AR `maxDets=100` |
| --- | ---: | ---: | ---: | ---: |
| Single-class multi-object detector | 0.162 | 0.454 | 0.072 | 0.421 |
| Multi-class multi-object detector | 0.095 | 0.274 | 0.034 | 0.289 |

Single-shark prototype validation summary:

| Metric | Value |
| --- | ---: |
| Classification accuracy | 0.8860 |
| Mean IoU | 0.1891 |
| Bounding-box accuracy | 0.0965 |
| Validation loss | 0.4420 |

### Example Predictions

Single-object shark detection:

![Single-object shark prediction](assets/examples/single_object_shark_prediction.png)

Single-class multi-object detection:

![Single-class multi-object metrics](assets/examples/single_class_multi_object_metrics.png)
![Single-class multi-object prediction](assets/examples/single_class_multi_object_prediction.png)

Multi-class multi-object detection:

![Multi-class multi-object metrics](assets/examples/multi_class_multi_object_metrics.png)
![Multi-class multi-object prediction](assets/examples/multi_class_multi_object_prediction.png)

External video inference frame:

![Video inference frame](assets/examples/video_inference_frame.png)

The trained model weights, raw datasets, and source videos are not committed because they are large and should stay outside GitHub.

Additional video links are tracked in [`assets/demo_video_links.md`](assets/demo_video_links.md).

## How To Run

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install .
```

Add the TensorFlow CSV dataset under `data/`, then convert the CSV annotations to COCO JSON:

```bash
python -m aquarium_detection.convert_roboflow_csv_to_coco \
  --data-root "data/Aquarium Combined.v2-raw-1024.tensorflow"
```

Train the final detector:

```bash
python -m aquarium_detection.train \
  --data-path "data/Aquarium Combined.v2-raw-1024.tensorflow" \
  --output-dir outputs/faster_rcnn_mobilenetv3 \
  --batch-size 4 \
  --num-workers 0 \
  --epochs 26 \
  --lr 0.0025
```

Evaluate a checkpoint on the held-out validation split:

```bash
python -m aquarium_detection.train \
  --data-path "data/Aquarium Combined.v2-raw-1024.tensorflow" \
  --resume outputs/faster_rcnn_mobilenetv3/best_model.pth \
  --output-dir outputs/evaluation \
  --test-only \
  --visualize
```

Run inference on an external video:

```bash
python -m aquarium_detection.infer_video \
  --checkpoint outputs/faster_rcnn_mobilenetv3/best_model.pth \
  --input data/videos/aquarium_input.mp4 \
  --output outputs/video_predictions.mp4 \
  --score-threshold 0.5
```

Run the MobileNetV3 ImageNet classification baseline:

```bash
python -m aquarium_detection.imagenet_baseline path/to/image.jpg
```

## Repository Structure

```text
aquarium_object_detection/
  README.md
  requirements.txt
  .gitignore
  pyproject.toml
  src/aquarium_detection/
  notebooks/
  assets/
  outputs/
  models/
  data/
```

## Limitations

- Dataset files are not included.
- Trained `.pth` model weights are not included.
- Source videos are linked, not committed.
- A reproducible final metric table should be added after a validated training run produces reliable detections.
- The final detector uses axis-aligned boxes only; oriented bounding boxes were not implemented.
- The legacy single-shark prototype is preserved for context but is not the primary final pipeline.

## Future Work

- Add a small public sample dataset or scripted dataset download step.
- Add final trained weights through GitHub Releases or another external artifact host.
- Add a reproducible metrics table after retraining produces a validated checkpoint.
- Add automated smoke tests for dataset loading and checkpoint inference.
- Extend to oriented bounding boxes if that detection format becomes useful for the dataset.
