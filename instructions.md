# YOLO11l Benchmark on UCF-Crime2Local

## Purpose

This project trains a **YOLO11 large object detection model** on the **UCF-Crime2Local** dataset and uses it as a benchmark against **VadCLIP**.

The point of this benchmark is not to prove YOLO is a better anomaly detector. The point is to create a strong, understandable comparison baseline based on **spatial detection**.

VadCLIP is an **anomaly detection model**. It tries to answer:

* is something abnormal happening
* when is it happening in the video

YOLO is an **object detection model**. It tries to answer:

* where is the important region in the frame
* how confident is the detector

Because these are different tasks, YOLO must be converted into an anomaly-scoring pipeline before it can be compared fairly to VadCLIP.

## Benchmark Goal

Train `YOLO11l` for **100 epochs** on the **Kaggle UCF-Crime2Local bounding box dataset**, then evaluate it in two ways:

1. **Detection performance**

   * mAP50
   * mAP50-95
   * precision
   * recall

2. **Anomaly performance**

   * frame-level ROC-AUC
   * frame-level Average Precision (AP)

The fair comparison to VadCLIP is the **anomaly performance**, not just YOLO mAP.

## Important Assumptions

This plan assumes:

* the Kaggle dataset contains videos or extracted frames
* the Kaggle dataset contains ground-truth bounding boxes
* the dataset is a subset of UCF-Crime with spatial annotations
* the goal is binary anomaly localization, not fine-grained crime classification

Use a **single-class detector** first:

* class `0 = anomaly_region`

This is the easiest and most defensible setup.

## High-Level Plan

1. Inspect the Kaggle dataset structure
2. Build a clean train/val/test split by video
3. Convert annotations into YOLO detection format
4. Train `yolo11l.pt` for 100 epochs
5. Run detection inference on the test set
6. Convert detection confidence into frame-level anomaly scores
7. Compare those anomaly scores against VadCLIP on the same videos

---

## Step 1: Inspect the Dataset

The agent should first inspect the Kaggle dataset and answer these questions:

* Are the inputs videos, frames, or both?
* Are bounding boxes already provided per frame?
* Are there train/test splits already?
* Are there normal and anomalous samples?
* Are annotations stored in CSV, JSON, TXT, XML, or another format?

The agent must produce a short summary of:

* input file structure
* annotation structure
* class mapping
* number of videos
* number of annotated frames
* whether normal frames have empty labels or separate metadata

## Step 2: Define the Training Task

Use **one class only** for the first version.

### Class Definition

* `0: anomaly_region`

This means YOLO is trained to detect the region in the frame where the abnormal event is happening.

Do not start with multiple crime classes unless there is a strong reason.

Why:

* easier labeling pipeline
* easier training
* cleaner comparison to VadCLIP
* less chance of label mismatch

## Step 3: Split by Video, Not by Frame

Create splits using **video IDs**, never frame IDs.

Recommended split:

* 70% train
* 15% validation
* 15% test

Rules:

* frames from the same video must never appear in different splits
* if Kaggle already has official train/test splits, preserve them
* create validation from the training portion only

The agent should save:

* `train_videos.txt`
* `val_videos.txt`
* `test_videos.txt`

## Step 4: Extract Frames if Needed

If the dataset contains videos rather than frame images, extract frames.

Recommended extraction rate:

* start with **2 fps**

Why:

* reduces near-duplicate frames
* keeps dataset size manageable
* still preserves temporal coverage

Example command:

```bash
ffmpeg -i input_video.mp4 -vf fps=2 output_frames/%06d.jpg
```

If the Kaggle dataset already includes frames, skip this step.

## Step 5: Convert Annotations to YOLO Format

YOLO detection expects this folder structure:

```text
dataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

For each image:

* there must be one matching label file with the same base name
* label files use YOLO normalized format

Each row in a label file must be:

```text
class x_center y_center width height
```

All values must be normalized to the image width and height.

If an image contains no anomaly box:

* create an empty label file

### Example

If an image has one anomaly box:

```text
0 0.512 0.441 0.210 0.330
```

The agent must write a converter that:

* reads Kaggle annotations
* maps all anomaly boxes to class `0`
* writes one `.txt` file per image
* validates coordinates are inside `[0, 1]`

## Step 6: Build `data.yaml`

Create a YOLO dataset config file named `data.yaml`.

```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val
test: images/test

names:
  0: anomaly_region
```

The agent must make sure this path is correct on the training machine.

## Step 7: Sanity Check the Converted Dataset

Before training, verify:

* every image has a matching label file
* there are no orphan label files
* images open correctly
* boxes are valid
* no label file has invalid class IDs
* train, val, and test splits do not overlap by video

Also generate a small preview set:

* 20 random training images with boxes drawn
* save them in `debug_samples/`

This step is mandatory.

## Step 8: Train YOLO11l for 100 Epochs

Use pretrained `YOLO11l`.

### Python version

```python
from ultralytics import YOLO

model = YOLO("yolo11l.pt")
results = model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    project="runs_ucfcrime",
    name="yolo11l_anomaly_region"
)
```

### CLI version

```bash
yolo train model=yolo11l.pt data=data.yaml epochs=100 imgsz=640 project=runs_ucfcrime name=yolo11l_anomaly_region
```

### Default training settings

Start with:

* `epochs=100`
* `imgsz=640`
* pretrained weights
* single class
* default optimizer and augmentation unless debugging shows problems

If GPU memory is too low:

* lower batch size first
* only lower image size if necessary

## Step 9: Save All Training Outputs

Keep these artifacts:

* best weights
* last weights
* training curves
* confusion matrix if available
* precision/recall plots
* mAP results
* run config

Expected output folder:

* `runs_ucfcrime/yolo11l_anomaly_region/`

## Step 10: Run Detection Inference on the Test Set

After training, run prediction on every test image.

The detector will output:

* boxes
* confidence scores
* class IDs

Save raw prediction results in a structured format such as:

* CSV
* JSONL
* parquet

Required fields:

* `video_id`
* `frame_id`
* `image_path`
* `num_boxes`
* `max_confidence`
* `all_confidences`
* `all_boxes`

## Step 11: Convert YOLO Outputs Into Anomaly Scores

This is the key comparison step.

VadCLIP outputs an anomaly score over time. YOLO does not. So YOLO predictions must be converted into a frame-level anomaly score.

### Rule for frame anomaly score

For each frame:

* if YOLO predicts one or more boxes, frame score = highest detection confidence
* if YOLO predicts no boxes, frame score = `0.0`

This gives one anomaly score per frame.

Optional smoothing:

* apply a short moving average across nearby frames
* keep both raw and smoothed scores

Save:

* `frame_scores_yolo.csv`

with columns:

* `video_id`
* `frame_id`
* `frame_path`
* `raw_score`
* `smoothed_score`

## Step 12: Evaluate YOLO as an Anomaly Detector

If frame-level anomaly ground truth exists for the test videos:

* align each predicted frame score with the true frame label
* compute frame-level ROC-AUC
* compute frame-level AP

If only clip-level labels exist:

* aggregate frame scores into clip scores
* use max or mean score per clip
* compute clip-level ROC-AUC and AP

Preferred aggregation:

* `max score` for anomaly-sensitive reporting
* optionally also report `mean score`

The final anomaly metrics should be saved in:

* `metrics_anomaly.json`

## Step 13: Evaluate YOLO as an Object Detector

Also report normal detection metrics from YOLO validation:

* precision
* recall
* mAP50
* mAP50-95

These are useful, but they are **not** the main comparison against VadCLIP.

Save them in:

* `metrics_detection.json`

## Step 14: Compare YOLO to VadCLIP Fairly

Do not compare:

* YOLO mAP
  to
* VadCLIP ROC-AUC

That is not a fair comparison.

The proper comparison is:

* YOLO frame-level ROC-AUC
* YOLO frame-level AP
* VadCLIP frame-level ROC-AUC
* VadCLIP frame-level AP

All four numbers must come from:

* the same videos
* the same test split
* the same frame alignment policy

### Comparison table format

| Model   | Type                       | Output                                        | ROC-AUC |   AP |
| ------- | -------------------------- | --------------------------------------------- | ------: | ---: |
| VadCLIP | anomaly detection          | frame anomaly score                           |    x.xx | x.xx |
| YOLO11l | detection-derived baseline | frame anomaly score from detection confidence |    x.xx | x.xx |

Optional second table:

| Model   | Precision | Recall | mAP50 | mAP50-95 |
| ------- | --------: | -----: | ----: | -------: |
| YOLO11l |      x.xx |   x.xx |  x.xx |     x.xx |

## Step 15: Create Visual Examples

For presentation and debugging, save examples from the test set showing:

* frame image
* predicted box
* predicted confidence
* true anomaly label
* frame anomaly score

Save:

* 5 correct positives
* 5 false positives
* 5 false negatives
* 5 correct negatives

This makes the benchmark much easier to explain.

## Recommended Project File Structure

```text
project/
  data/
    raw/
    processed/
    splits/
  scripts/
    inspect_dataset.py
    extract_frames.py
    convert_to_yolo.py
    validate_yolo_dataset.py
    train_yolo11l.py
    infer_yolo.py
    build_frame_scores.py
    evaluate_detection.py
    evaluate_anomaly.py
    compare_with_vadclip.py
  outputs/
    debug_samples/
    predictions/
    metrics/
    figures/
  runs_ucfcrime/
```

## Required Scripts

### `inspect_dataset.py`

Purpose:

* summarize Kaggle dataset structure
* detect annotation format
* count videos, frames, and labels

### `extract_frames.py`

Purpose:

* extract frames from videos if needed
* preserve video ID to frame mapping

### `convert_to_yolo.py`

Purpose:

* convert Kaggle annotations to YOLO label files
* map all boxes to class `0`

### `validate_yolo_dataset.py`

Purpose:

* verify image-label consistency
* verify boxes are valid
* generate preview images with boxes

### `train_yolo11l.py`

Purpose:

* train YOLO11l for 100 epochs on the converted dataset

### `infer_yolo.py`

Purpose:

* run trained detector on test images
* save raw prediction outputs

### `build_frame_scores.py`

Purpose:

* convert YOLO detections into one anomaly score per frame

### `evaluate_detection.py`

Purpose:

* compute precision, recall, mAP50, mAP50-95

### `evaluate_anomaly.py`

Purpose:

* align frame scores with anomaly labels
* compute ROC-AUC and AP

### `compare_with_vadclip.py`

Purpose:

* load YOLO anomaly metrics
* load VadCLIP anomaly metrics
* produce side-by-side comparison tables and plots

## Acceptance Criteria

The implementation is successful only if all of these are true:

1. The Kaggle dataset is converted to valid YOLO format
2. No video leakage exists between train, val, and test
3. YOLO11l trains for 100 epochs without crashing
4. Best model weights are saved
5. Detection metrics are produced
6. Frame-level anomaly scores are produced from YOLO outputs
7. YOLO anomaly metrics are computed
8. YOLO and VadCLIP are compared on the same test videos
9. Example visualizations are saved
10. All scripts are reproducible from the command line

## Notes for the Agent

* Start simple
* Do not add tracking in version 1
* Do not add multi-class crime labels in version 1
* Do not compare mAP directly to VadCLIP AUC
* Keep the whole pipeline reproducible and easy to explain

## Short Explanation for Humans

This benchmark uses YOLO11l as a spatial baseline. YOLO learns to find the abnormal region in annotated UCF-Crime frames. Its detection confidence is then converted into a frame-level anomaly score. That anomaly score is compared directly against VadCLIP on the same test videos using ROC-AUC and AP.

If you want, I can also turn this into a tighter **agent-ready prompt** with direct implementation commands and expected outputs.
