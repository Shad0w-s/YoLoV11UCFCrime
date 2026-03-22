# YOLO11l on UCF-Crime2Local

Train **YOLO11 large** on the UCF-Crime2Local frame dataset (single class `anomaly_region`), run test inference, and export detection metrics plus frame-level anomaly scores derived from max detection confidence.

This repo focuses on the **YOLO** pipeline only. VadCLIP lives in a separate project; optional **merge** of exported JSON metrics is supported here.

## Prerequisites

- Python 3.10+ recommended (3.14 may work if PyTorch wheels exist for your platform).
- Dataset root containing `rgb-images/`, `labels/`, `train.txt`, `test.txt`, `labels.txt`.

## Setup

```bash
cd /path/to/YoLoV11UCFCrime
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Optional: reproducible pins from a known-good environment
# pip install -r requirements.lock.txt
```

## Environment

| Variable | Description |
|----------|-------------|
| `UCFCRIME2LOCAL_ROOT` | Absolute path to the UCF-Crime2Local folder (required for most scripts). |

Example:

```bash
export UCFCRIME2LOCAL_ROOT=/Users/you/Downloads/ucfcrime2local
```

## Pipeline (CLI)

Run from the repository root with the venv activated.

1. **Inspect dataset**

   ```bash
   python scripts/inspect_dataset.py --out outputs/dataset_summary.md
   ```

2. **Split videos** (preserves official `test.txt`; train/val from `train.txt` only, ~85% / ~15% by video)

   ```bash
   python scripts/split_videos.py --seed 42
   ```

3. **Convert to YOLO layout** (`data/processed/` + `data/processed/data.yaml`)

   ```bash
   python scripts/convert_to_yolo.py
   ```

4. **Validate + debug previews** (20 overlays → `outputs/debug_samples/`)

   ```bash
   python scripts/validate_yolo_dataset.py --num-samples 20
   ```

5. **Train YOLO11l** (100 epochs, imgsz 640, `runs_ucfcrime/yolo11l_anomaly_region/`)

   ```bash
   python scripts/train_yolo11l.py --batch 16 --device mps
   ```

   Use `--device 0` for CUDA, `cpu` for CPU.

6. **Inference on test set**

   ```bash
   python scripts/infer_yolo.py --weights runs_ucfcrime/yolo11l_anomaly_region/weights/best.pt
   ```

   Smoke test (first *N* images, e.g. pretrained `yolo11l.pt`):

   ```bash
   python scripts/infer_yolo.py --weights yolo11l.pt --max-images 50 --device mps
   ```

7. **Frame anomaly scores**

   ```bash
   python scripts/build_frame_scores.py --smooth-window 5
   ```

8. **Detection metrics** (from `model.val()`)

   ```bash
   python scripts/evaluate_detection.py --weights runs_ucfcrime/yolo11l_anomaly_region/weights/best.pt
   ```

9. **Anomaly metrics** (optional; needs frame binary labels CSV)

   ```bash
   python scripts/evaluate_anomaly.py --frame-labels path/to/frame_labels.csv
   ```

   CSV columns: `frame_path` or (`video_id`, `frame_id`), `label` (0/1).

10. **Visualization examples** (optional labels for TP/FP/FN/TN)

    ```bash
    python scripts/visualize_examples.py --weights runs_ucfcrime/yolo11l_anomaly_region/weights/best.pt
    ```

11. **Optional: merge YOLO + VadCLIP JSON** (no VadCLIP code)

    ```bash
    python scripts/merge_metrics_optional.py --vadclip path/to/vadclip_metrics.json
    ```

    `vadclip_metrics.json` example (from your other project):

    ```json
    { "roc_auc": 0.85, "average_precision": 0.42 }
    ```

    Requires `outputs/metrics/metrics_anomaly.json` from step 9 with real frame labels.

## Outputs

| Path | Content |
|------|---------|
| `data/splits/` | `train_videos.txt`, `val_videos.txt`, `test_videos.txt` |
| `data/processed/` | YOLO `images/`, `labels/`, `data.yaml` |
| `outputs/debug_samples/` | Validation preview images |
| `outputs/predictions/` | `test_predictions.jsonl`, `frame_scores_yolo.csv` |
| `outputs/metrics/` | `metrics_detection.json`, `metrics_anomaly.json` (if labels provided) |
| `outputs/figures/` | Example qualitative figures |
| `runs_ucfcrime/` | Ultralytics training runs |


