# Data

Place datasets here. The local project data is the Roboflow Aquarium object-detection export in TensorFlow CSV format:

```text
data/Aquarium Combined.v2-raw-1024.tensorflow/
  train/_annotations.csv
  valid/_annotations.csv
  test/_annotations.csv
```

Convert the CSV annotations to COCO JSON before training:

```bash
python -m aquarium_detection.convert_roboflow_csv_to_coco \
  --data-root "data/Aquarium Combined.v2-raw-1024.tensorflow"
```

The converter writes `_annotations.coco.json` into each split folder. Dataset images and generated annotation files are ignored by git.
