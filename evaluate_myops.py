"""Evaluation script for TransUNet on MyoPS380 test set.

Protocol: myops380_voxel_v3 (Shared Evaluator)
Outputs:
  1) <run_dir>/test_predictions/<CaseID>_pred.npz (Canonical label format, label 3 is Scar)
  2) <output_dir>/per_case.csv (Full per-case, per-region breakdown)
  3) <output_dir>/metrics.json (Official benchmark summary matching SCAR schema)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.ndimage import zoom
import torch

from datasets.dataset_myops import MyopsVolumeDataset, canonicalize_label
from networks.vit_seg_modeling import VisionTransformer, CONFIGS
from utils.metrics import BENCHMARK_PROTOCOL, dataset_rows, summarize_dataset_rows

logger = logging.getLogger("TransUNet_Eval")


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return device


def json_safe(val: Any) -> Any:
    if isinstance(val, dict):
        return {str(k): json_safe(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [json_safe(v) for v in val]
    if isinstance(val, (float, np.floating)):
        return float(val) if math.isfinite(val) else None
    if isinstance(val, np.generic):
        return val.item()
    return val


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TransUNet on MyoPS380 official 76 test cases")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint (e.g. outputs/runs/transunet_seed1234/checkpoints/transunet_seed1234.pt)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=os.environ.get("LOCAL_DATASET", "E:/STUDY/DATASET/MyoPS380/Processed_data"),
        help="Path to preprocessed MyoPS380 dataset root containing bSSFP, LGE, T2w",
    )
    parser.add_argument(
        "--list-dir",
        type=str,
        default="lists/lists_MyoPS380",
        help="Path to split manifests folder containing test_vol.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation reports and prediction files",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        default=True,
        help="Save test_predictions/<CaseID>_pred.npz files (default: True)",
    )
    parser.add_argument("--device", type=str, default="auto", help="Execution device ('auto', 'cuda', 'cpu')")
    return parser.parse_args()


def predict_case_volume(
    model: torch.nn.Module,
    img_bssfp: np.ndarray,
    img_lge: np.ndarray,
    img_t2w: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Predict a 3D volume by running 2D slice inference and stacking along depth (H, W, D)."""
    h, w, d = img_bssfp.shape
    prediction_3d = np.zeros((h, w, d), dtype=np.uint8)

    model.eval()
    with torch.no_grad():
        for slice_idx in range(d):
            s0 = img_bssfp[:, :, slice_idx]
            s1 = img_lge[:, :, slice_idx]
            s2 = img_t2w[:, :, slice_idx]

            # Zoom to 128x128 if needed
            if h != 128 or w != 128:
                s0 = zoom(s0, (128 / h, 128 / w), order=3)
                s1 = zoom(s1, (128 / h, 128 / w), order=3)
                s2 = zoom(s2, (128 / h, 128 / w), order=3)

            stacked = np.stack([s0, s1, s2], axis=0)  # (3, 128, 128)
            tensor = torch.from_numpy(stacked).unsqueeze(0).float().to(device)  # (1, 3, 128, 128)

            logits = model(tensor)  # (1, 4, 128, 128)
            pred_slice = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            if h != 128 or w != 128:
                pred_slice = zoom(pred_slice, (h / 128, w / 128), order=0)

            prediction_3d[:, :, slice_idx] = pred_slice

    return prediction_3d


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    # Default output directory: <run_dir>/evaluation
    run_dir = ckpt_path.parent.parent
    output_dir = Path(args.output_dir) if args.output_dir else (run_dir / "evaluation")
    pred_dir = run_dir / "test_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "evaluate.log", encoding="utf-8"),
        ],
    )

    logger.info("================================================================================")
    logger.info("TransUNet MyoPS380 Evaluation - Protocol: myops380_voxel_v3")
    logger.info("Checkpoint: %s", ckpt_path)
    logger.info("Output directory: %s", output_dir)
    logger.info("Prediction directory: %s", pred_dir)
    logger.info("================================================================================")

    device = resolve_device(args.device)
    logger.info("Using device: %s", device)

    # Initialize model
    config = CONFIGS["R50-ViT-B_16"]
    config.patches.grid = (8, 8)
    config.n_classes = 4
    model = VisionTransformer(config, img_size=128, num_classes=4, zero_head=False, vis=False)

    # Load weights
    logger.info("Loading checkpoint weights...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully. Trainable params: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    # Load volume dataset
    dataset = MyopsVolumeDataset(
        data_root=args.data_root,
        list_dir=args.list_dir,
        split="test_vol",
        label_order="legacy",
    )
    logger.info("Total test volume cases: %d", len(dataset))

    all_metric_rows: list[dict[str, Any]] = []
    has_labels = False

    eval_start_time = time.time()

    for idx in range(len(dataset)):
        case_sample = dataset[idx]
        case_name = case_sample["case_name"]
        img_bssfp = case_sample["image"]
        img_lge = case_sample["image1"]
        img_t2w = case_sample["image2"]
        target = case_sample["label"]

        # Run 3D volume inference
        pred_3d = predict_case_volume(model, img_bssfp, img_lge, img_t2w, device)

        # Save test prediction in canonical format
        if args.save_predictions:
            pred_file = pred_dir / f"{case_name}_pred.npz"
            np.savez_compressed(pred_file, prediction=pred_3d.astype(np.uint8))

        # Compute metrics if ground truth is available
        if target is not None:
            has_labels = True
            rows = dataset_rows(pred_3d, target, case_name, compute_distance=True)
            all_metric_rows.extend(rows)

            scar_row = next((r for r in rows if r["region"] == "scar"), None)
            edema_inc_row = next((r for r in rows if r["region"] == "edema_inclusive"), None)
            s_dice = f"{scar_row['dice']:.4f}" if (scar_row and scar_row["dice"] is not None) else "None"
            e_dice = f"{edema_inc_row['dice']:.4f}" if (edema_inc_row and edema_inc_row["dice"] is not None) else "None"
            logger.info("Case [%2d/%2d] %s | Scar Dice: %s | Edema-Inc Dice: %s", idx + 1, len(dataset), case_name, s_dice, e_dice)
        else:
            logger.info("Case [%2d/%2d] %s | Prediction saved (No GT label available)", idx + 1, len(dataset), case_name)

    total_time = time.time() - eval_start_time
    logger.info("Inference complete in %.1f seconds.", total_time)

    if has_labels and all_metric_rows:
        # Write per_case.csv
        csv_file = output_dir / "per_case.csv"
        fieldnames = [
            "case",
            "region",
            "dice",
            "iou",
            "precision",
            "recall",
            "hd95_voxel",
            "asd_voxel",
            "status",
            "prediction_voxels",
            "target_voxels",
            "hd95_unit",
        ]
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in all_metric_rows:
                writer.writerow(row)
        logger.info("Per-case metrics saved to: %s", csv_file)

        # Summarize rows and write metrics.json
        summary = summarize_dataset_rows(all_metric_rows)
        metrics_payload = {
            "protocol": BENCHMARK_PROTOCOL,
            "checkpoint": str(ckpt_path),
            "total_cases": len(dataset),
            "summary": summary,
        }
        json_file = output_dir / "metrics.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(json_safe(metrics_payload), f, indent=2)
        logger.info("Summary metrics saved to: %s", json_file)

        logger.info("================================================================================")
        logger.info("BENCHMARK RESULTS SUMMARY (Protocol: myops380_voxel_v3)")
        logger.info("Avg Pathology Dice (Scar & Edema-Excl):        %.4f", summary.get("avg_pathology_dice") or 0.0)
        logger.info("Avg Pathology Inclusive Dice (Scar & Edema-Inc): %.4f", summary.get("avg_pathology_inclusive_dice") or 0.0)
        for r_name in ("scar", "edema", "edema_inclusive", "normal_myocardium", "myocardial_ring"):
            reg = summary.get(r_name, {})
            m_dice = reg.get("mean_dice")
            m_hd95 = reg.get("mean_hd95_voxel")
            logger.info("  Region: %-18s | Mean Dice: %s | Mean HD95 (vox): %s", r_name, f"{m_dice:.4f}" if m_dice else "N/A", f"{m_hd95:.2f}" if m_hd95 else "N/A")
        logger.info("================================================================================")
    else:
        logger.info("Predictions saved without ground-truth comparison.")


if __name__ == "__main__":
    main()
