"""Unit and CLI integration tests for TransUNet efficiency benchmarking."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from networks.vit_seg_modeling import CONFIGS, VisionTransformer
from utils.efficiency import (
    BYTES_PER_GB,
    BYTES_PER_MB,
    build_efficiency_payload,
    check_rule_compliance,
    count_parameters,
    format_efficiency_report,
    measure_flops,
    measure_inference_efficiency,
    summarize_training_efficiency,
)


class EfficiencyBenchmarkTests(unittest.TestCase):
    def test_transunet_complexity_and_inference_cpu(self) -> None:
        cfg = CONFIGS["R50-ViT-B_16"]
        cfg.patches.grid = (8, 8)
        cfg.n_classes = 4
        model = VisionTransformer(cfg, img_size=128, num_classes=4, zero_head=False, vis=False).float()

        params_info = count_parameters(model)
        self.assertEqual(params_info["total_params"], 105_174_980)
        self.assertEqual(params_info["trainable_params"], 105_174_980)
        self.assertEqual(params_info["trainable_params_excel"], "105.17M")

        flops_info = measure_flops(model, input_shape=(1, 3, 128, 128), device=torch.device("cpu"))
        self.assertGreater(flops_info["flops_total"], 18_000_000_000)
        self.assertGreater(flops_info["flops_g"], 18.0)
        self.assertLess(flops_info["flops_g"], 20.0)

        inf_info = measure_inference_efficiency(
            model, input_shape=(1, 3, 128, 128), device=torch.device("cpu"), warmup=2, repeats=3
        )
        self.assertGreater(inf_info["latency_ms_per_slice"], 0.0)
        self.assertEqual(inf_info["batch_size"], 1)
        self.assertEqual(inf_info["peak_vram_mb"], 0.0)
        self.assertEqual(inf_info["peak_vram_gb"], 0.0)

    def test_summarize_training_efficiency_skips_first_5_epochs(self) -> None:
        epoch_times = [100.0] * 5 + [10.0] * 295
        peak_bytes = int(4096 * BYTES_PER_MB)
        summary = summarize_training_efficiency(
            epoch_times_s=epoch_times,
            total_wall_clock_s=sum(epoch_times) + 120.0,
            peak_vram_bytes=peak_bytes,
            planned_epochs=300,
            warmup_epochs=5,
            peak_reserved_bytes=int(4.5 * BYTES_PER_GB),
        )
        self.assertEqual(summary["warmup_epochs_skipped"], 5)
        self.assertEqual(summary["measured_epoch_range"], "6..300")
        self.assertEqual(summary["time_per_epoch_s"], 10.0)
        self.assertAlmostEqual(summary["total_time_hours"], sum(epoch_times) / 3600.0, places=3)
        self.assertEqual(summary["peak_vram_mb"], 4096.0)
        self.assertEqual(summary["peak_vram_gb"], 4.0)
        self.assertEqual(summary["peak_vram_reserved_mb"], 4608.0)
        self.assertEqual(summary["peak_vram_reserved_gb"], 4.5)

    def test_rule_compliance_flags(self) -> None:
        ok, warnings, hw = check_rule_compliance(
            device=torch.device("cpu"),
            amp_mode="none",
            num_workers=0,
            pin_memory=True,
            batch_size=16,
            planned_epochs=300,
        )
        self.assertFalse(ok)
        self.assertTrue(any("CPU" in w for w in warnings))
        self.assertEqual(hw["precision"], "fp32")

        ok2, warnings2, _ = check_rule_compliance(
            device=torch.device("cpu"),
            amp_mode="auto",
            num_workers=4,
            pin_memory=True,
            batch_size=8,
            planned_epochs=10,
        )
        self.assertFalse(ok2)
        self.assertGreaterEqual(len(warnings2), 5)

    def test_cli_train_myops_end_to_end_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_root = tmp_path / "Processed_data"
            list_dir = tmp_path / "lists"
            run_root = tmp_path / "runs"
            eff_out = tmp_path / "eff" / "efficiency_metrics.json"
            list_dir.mkdir(parents=True)

            for mod in ("bSSFP", "LGE", "T2w"):
                (data_root / mod / "train_npz").mkdir(parents=True)

            train_lines: list[str] = []
            for idx in range(2):
                name = f"case9999_slice{idx:03d}"
                train_lines.append(name)
                label = np.zeros((128, 128), dtype=np.uint8)
                label[40:80, 40:80] = 1
                label[50:65, 50:65] = 2
                label[65:75, 50:65] = 3
                for mod in ("bSSFP", "LGE", "T2w"):
                    img = np.random.rand(128, 128).astype(np.float32)
                    np.savez(data_root / mod / "train_npz" / f"{name}.npz", image=img, label=label)

            (list_dir / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")

            # 1) Test --efficiency-only
            cmd_only = [
                sys.executable,
                str(REPO_ROOT / "train_myops.py"),
                "--data-root",
                str(data_root),
                "--list-dir",
                str(list_dir),
                "--run-root",
                str(run_root),
                "--run-name",
                "smoke_eff_only",
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--amp",
                "none",
                "--num-workers",
                "0",
                "--device",
                "cpu",
                "--efficiency-output",
                str(eff_out),
                "--efficiency-only",
            ]
            res = subprocess.run(cmd_only, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, msg=res.stderr)
            self.assertTrue(eff_out.is_file())
            payload = json.loads(eff_out.read_text(encoding="utf-8"))
            self.assertEqual(payload["excel_summary"]["Trainable Params (M)"], 105.17)
            self.assertGreater(payload["excel_summary"]["FLOPs (G)"], 18.0)
            self.assertIsNone(payload["training"])

            # 2) Test full 2-epoch training with --amp none --num-workers 0
            eff_out_train = tmp_path / "eff_train" / "efficiency_metrics.json"
            base_cmd = [
                sys.executable,
                str(REPO_ROOT / "train_myops.py"),
                "--data-root",
                str(data_root),
                "--list-dir",
                str(list_dir),
                "--run-root",
                str(run_root),
                "--run-name",
                "smoke_train",
                "--batch-size",
                "2",
                "--epochs",
                "2",
                "--amp",
                "none",
                "--num-workers",
                "0",
                "--device",
                "cpu",
                "--efficiency-output",
                str(eff_out_train),
            ]
            r1 = subprocess.run(base_cmd, capture_output=True, text=True)
            self.assertEqual(r1.returncode, 0, msg=r1.stderr)
            ckpt_file = run_root / "smoke_train" / "checkpoints" / "transunet_seed1234.pt"
            self.assertTrue(ckpt_file.is_file())
            self.assertTrue(eff_out_train.is_file())
            train_payload = json.loads(eff_out_train.read_text(encoding="utf-8"))
            self.assertEqual(train_payload["training"]["completed_epochs"], 2)
            self.assertEqual(len(train_payload["training"]["all_epoch_times_s"]), 2)
            self.assertTrue((run_root / "smoke_train" / "efficiency_metrics.json").is_file())
            self.assertTrue((run_root / "smoke_train" / "logs" / "efficiency_summary.txt").is_file())

            # 3) Simulate mid-run interruption at epoch 1 and verify --resume preserves epoch_times
            ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            first_epoch_time = ckpt["epoch_times"][0]
            ckpt["epoch"] = 1
            ckpt["completed_epochs"] = 1
            ckpt["training_complete"] = False
            ckpt["epoch_times"] = [first_epoch_time]
            torch.save(ckpt, ckpt_file)
            eff_out_train.unlink()

            r2 = subprocess.run(base_cmd + ["--resume", str(ckpt_file)], capture_output=True, text=True)
            self.assertEqual(r2.returncode, 0, msg=r2.stderr)
            self.assertTrue(eff_out_train.is_file())
            resumed_payload = json.loads(eff_out_train.read_text(encoding="utf-8"))
            self.assertEqual(resumed_payload["training"]["completed_epochs"], 2)
            self.assertEqual(len(resumed_payload["training"]["all_epoch_times_s"]), 2)
            self.assertAlmostEqual(
                resumed_payload["training"]["all_epoch_times_s"][0],
                round(first_epoch_time, 4),
                places=4,
            )

    def test_format_efficiency_report_contains_both_units(self) -> None:
        payload = build_efficiency_payload(
            model_name="TransUNet",
            model_id="transunet",
            seed=1234,
            params_info={
                "total_params": 105174980,
                "trainable_params": 105174980,
                "total_params_m": 105.175,
                "trainable_params_m": 105.175,
                "trainable_params_excel": "105.17M",
                "total_params_excel": "105.17M",
            },
            flops_info={"flops_total": 18738020352, "flops_g": 18.738},
            inference_info={
                "latency_ms_per_slice": 5.12,
                "warmup_runs": 30,
                "measure_runs": 100,
                "peak_vram_mb": 512.0,
                "peak_vram_gb": 0.5,
                "peak_vram_reserved_mb": 600.0,
            },
            training_info={
                "planned_epochs": 300,
                "completed_epochs": 300,
                "warmup_epochs_skipped": 5,
                "measured_epoch_range": "6..300",
                "time_per_epoch_s": 12.34,
                "total_time_seconds": 3702.0,
                "total_time_hours": 1.0283,
                "total_wall_clock_hours": 1.05,
                "peak_vram_mb": 4096.0,
                "peak_vram_gb": 4.0,
                "peak_vram_reserved_mb": 4500.0,
            },
            rule_compliant=True,
            compliance_warnings=[],
            hardware_info={
                "gpu_name": "NVIDIA A100-SXM4-40GB",
                "visible_gpu_count": 1,
                "pytorch_version": "2.4.0",
                "cuda_version": "12.1",
                "precision": "fp32",
                "amp_mode": "none",
                "num_workers": 0,
                "pin_memory": True,
                "batch_size": 16,
            },
        )
        txt = format_efficiency_report(payload)
        self.assertIn("4096.00 MB (4.0000 GB)", txt)
        self.assertIn("512.00 MB (0.5000 GB)", txt)
        self.assertIn("105.17M", txt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
