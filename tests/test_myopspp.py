"""Label/geometry regression checks: python -m unittest discover -s tests -v."""
import copy
import json
import logging
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk
import torch

from datasets.dataset_myops import MyopsDataset, RandomGenerator, canonicalize_label
from datasets.dataset_myopspp import (
    MyopsPPDataset, MyopsPPVolumeDataset, map_myopspp_label,
    resolve_dataset_format, validate_data_checkpoint,
)
from utils import DiceLoss
from utils.metrics import dataset_rows
import train_myops
from train_myops import TRAIN_AUGMENTATION_PROTOCOL, build_train_transform, validate_augmentation_checkpoint


def write_nifti(path, xyz, spacing=(1.0, 1.0, 7.0)):
    image = sitk.GetImageFromArray(xyz.transpose(2, 1, 0))
    image.SetSpacing(spacing)
    sitk.WriteImage(image, str(path))


class LabelContractTests(unittest.TestCase):
    def test_raw_mapping_and_legacy_mapping_are_distinct(self):
        np.testing.assert_array_equal(
            map_myopspp_label(np.array([0, 200, 500, 600, 1220, 2221], dtype=np.uint16)),
            [0, 1, 0, 0, 2, 3],
        )
        np.testing.assert_array_equal(canonicalize_label(np.array([0, 1, 2, 3])), [0, 1, 3, 2])

    def test_reject_invalid_or_already_canonical_labels(self):
        for values in ([1, 2, 3], [1221], [200.5], [-1], [np.nan], [np.inf], []):
            with self.subTest(values=values), self.assertRaises(ValueError):
                map_myopspp_label(np.asarray(values))

    def test_ce_dice_and_metrics_agree_on_edema_and_scar(self):
        raw = np.array([[0, 200, 500], [600, 1220, 2221]], dtype=np.uint16)
        canonical = map_myopspp_label(raw)
        target = torch.from_numpy(canonical.astype(np.int64)).unsqueeze(0)
        logits = torch.full((1, 4, 2, 3), -20.0).scatter_(1, target.unsqueeze(1), 20.0)
        ce = torch.nn.CrossEntropyLoss()
        dice = DiceLoss(4)
        self.assertLess(ce(logits, target).item(), 1e-6)
        self.assertLess(dice(logits, target, softmax=True).item(), 1e-6)
        swapped_logits = logits[:, [0, 1, 3, 2]]
        self.assertGreater(ce(swapped_logits, target).item(), 1)
        self.assertGreater(dice(swapped_logits, target, softmax=True).item(), 0.4)
        rows = dataset_rows(canonical[..., None], canonical[..., None], "case", compute_distance=False)
        for row in rows:
            self.assertEqual(row["dice"], 1.0)
        # Check metric semantics independently of the adapter's output.
        prediction = np.zeros_like(canonical)
        prediction[raw == 1220] = 2
        prediction[raw == 2221] = 3
        rows = dataset_rows(prediction[..., None], canonical[..., None], "case", compute_distance=False)
        for row in rows:
            if row["region"] in ("edema", "scar", "edema_inclusive"):
                self.assertEqual(row["dice"], 1.0)

    def test_geometric_augmentation_keeps_channels_and_labels_aligned(self):
        label = np.tile(np.arange(4, dtype=np.uint8), (128, 32))
        sample = {k: label.astype(np.float32) / 3 for k in ("image", "image1", "image2")}
        sample["label"] = label
        with patch("datasets.dataset_myops.random.random", side_effect=[0.9, 0, 0, 0, 0]):
            result = RandomGenerator(intensity_max=1)(sample)
        for channel in result["image"]:
            torch.testing.assert_close(channel * 3, result["label"].float())
        self.assertEqual(result["label"].dtype, torch.long)
        self.assertEqual(torch.unique(result["label"]).tolist(), [0, 1, 2, 3])

    def test_gamma_uses_selected_intensity_scale(self):
        for maximum in (1.0, 255.0):
            sample = {k: np.full((128, 128), 0.25, np.float32) for k in ("image", "image1", "image2")}
            sample["label"] = np.full((128, 128), 2, np.uint8)
            with patch("datasets.dataset_myops.random.random", side_effect=[0.9, 0, 0, 0.9, 0]), patch("datasets.dataset_myops.random.uniform", return_value=0.8):
                result = RandomGenerator(intensity_max=maximum)(sample)
            expected = (0.25 / maximum) ** 0.8 * maximum
            torch.testing.assert_close(result["image"], torch.full((3, 128, 128), expected))
            self.assertTrue((result["label"] == 2).all())


class TrainingAugmentationTests(unittest.TestCase):
    def test_both_training_branches_use_the_shared_transform(self):
        class DatasetCaptured(Exception):
            pass

        for dataset, constructor in [("myops380", "MyopsDataset"), ("myopspp", "MyopsPPDataset")]:
            args = SimpleNamespace(epochs=1, batch_size=1, num_workers=0, resume=None,
                                   data_root="unused", list_dir="unused", dataset=dataset)
            transform = build_train_transform()
            with self.subTest(dataset=dataset), patch("train_myops.parse_args", return_value=args), \
                 patch("train_myops.build_train_transform", return_value=transform) as factory, \
                 patch(f"train_myops.{constructor}", side_effect=DatasetCaptured) as loader:
                with self.assertRaises(DatasetCaptured):
                    train_myops.main()
                factory.assert_called_once_with()
                self.assertIs(loader.call_args.kwargs["transform"], transform)
                self.assertEqual(transform.intensity_max, 255.0)
                self.assertEqual(transform.output_size, (128, 128))

    def test_shared_augmentation_checkpoint_is_supported_for_both_datasets(self):
        for dataset in ("myops380", "myopspp"):
            validate_augmentation_checkpoint({"augmentation_protocol": copy.deepcopy(TRAIN_AUGMENTATION_PROTOCOL)}, dataset)

    def test_original_myops380_checkpoint_remains_supported(self):
        validate_augmentation_checkpoint({"training_protocol": {"cohort": "myops380"}}, "myops380")

    def test_older_unit_scale_myopspp_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "augmentation differs"):
            validate_augmentation_checkpoint({"training_protocol": {"cohort": "myopspp"}}, "myopspp")
        for dataset in ("myops380", "myopspp"):
            for key, value in [("intensity_max", 1.0), ("post_resize_clip", True)]:
                incompatible = copy.deepcopy(TRAIN_AUGMENTATION_PROTOCOL)
                incompatible[key] = value
                with self.subTest(dataset=dataset, key=key), self.assertRaisesRegex(ValueError, "augmentation differs"):
                    validate_augmentation_checkpoint({"augmentation_protocol": incompatible}, dataset)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = {
            "dataset_id": "test_roi128", "split_protocol": "no_val",
            "patients": {"train": ["Case2001"], "test_vol": ["Case3001"]},
            "patient_counts": {"train": 1, "test_vol": 1},
        }
        self.save_split()
        x, y, z = np.indices((128, 128, 2))
        self.images = [((x + 2*y + 3*z)/512 + offset).astype(np.float32) for offset in (0, 0.1, 0.2)]
        self.raw = np.zeros((128, 128, 2), dtype=np.uint16)
        self.raw[:6, 0, :] = np.array([0, 200, 500, 600, 1220, 2221])[:, None]
        for center, case in [("CenterB", "Case2001"), ("CenterC", "Case3001")]:
            folder = self.root / "cases" / center / case
            folder.mkdir(parents=True)
            for tag, array in zip(("C0", "LGE", "T2", "gd"), [*self.images, self.raw]):
                write_nifti(folder / f"{case}_{tag}.nii.gz", array)
        self.train_folder = self.root / "cases" / "CenterB" / "Case2001"

    def save_split(self):
        (self.root / "splits.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_train_channel_axis_and_label_order(self):
        data = MyopsPPDataset(self.root)
        self.assertEqual(resolve_dataset_format(self.root), "myopspp")
        self.assertEqual(len(data), 2)
        self.assertTrue(all(name.startswith("Case2001_slice") for name in data.sample_list))
        sample = data[1]
        self.assertEqual(sample["image"].dtype, torch.float32)
        self.assertEqual(sample["label"].dtype, torch.long)
        np.testing.assert_array_equal(sample["image"].numpy(), np.stack([a[:, :, 1] for a in self.images]))
        np.testing.assert_array_equal(sample["label"][:6, 0].numpy(), [0, 1, 0, 0, 2, 3])

    def test_test_volume_matches_training_axes_and_class_contract(self):
        data = MyopsPPVolumeDataset(self.root)
        self.assertEqual(len(data), 1)
        sample = data[0]
        self.assertEqual(sample["case_name"], "Case3001")
        for key, expected in zip(("image", "image1", "image2"), self.images):
            np.testing.assert_array_equal(sample[key], expected)
        np.testing.assert_array_equal(sample["label"][:6, 0, 0], [0, 1, 0, 0, 2, 3])
        np.testing.assert_array_equal(sample["spacing"], [1, 1, 7])
        np.testing.assert_array_equal(sample["affine"], np.diag([-1, -1, 7, 1]))
        self.assertEqual(data.data_protocol, MyopsPPDataset(self.root).data_protocol)

    def test_reject_overlapping_patients(self):
        self.manifest["patients"]["test_vol"] = ["Case2001"]
        self.save_split()
        with self.assertRaisesRegex(ValueError, "overlapping"):
            MyopsPPDataset(self.root)

    def test_reject_missing_modality(self):
        (self.train_folder / "Case2001_T2.nii.gz").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "T2"):
            MyopsPPDataset(self.root)

    def test_reject_geometry_mismatch(self):
        write_nifti(self.train_folder / "Case2001_LGE.nii.gz", self.images[1], spacing=(2, 1, 7))
        with self.assertRaisesRegex(ValueError, "Unaligned"):
            MyopsPPDataset(self.root)

    def test_reject_unnormalized_input(self):
        write_nifti(self.train_folder / "Case2001_LGE.nii.gz", self.images[1] * 255)
        data = MyopsPPDataset(self.root)
        with self.assertRaisesRegex(ValueError, "normalized"):
            data[0]

    def test_reject_unknown_raw_mask_ids(self):
        self.raw[0, 0, 0] = 1234
        write_nifti(self.train_folder / "Case2001_gd.nii.gz", self.raw)
        with self.assertRaisesRegex(ValueError, "raw MyoPS"):
            MyopsPPDataset(self.root)[0]

    def test_checkpoint_cannot_cross_datasets_or_swap_classes(self):
        expected = MyopsPPDataset(self.root).data_protocol
        validate_data_checkpoint({"data_protocol": copy.deepcopy(expected)}, expected)
        validate_data_checkpoint({}, None)  # Preserve old MyoPS380 checkpoints.
        wrong = copy.deepcopy(expected)
        wrong["raw_to_canonical"].update({"1220": 3, "2221": 2})
        for checkpoint, contract in [({}, expected), ({"data_protocol": wrong}, expected), ({"data_protocol": expected}, None)]:
            with self.subTest(checkpoint=checkpoint), self.assertRaisesRegex(ValueError, "mapping differs"):
                validate_data_checkpoint(checkpoint, contract)
        changed_split = copy.deepcopy(expected)
        changed_split["split_sha256"] = "another_split"
        with self.assertRaises(ValueError):
            validate_data_checkpoint({"data_protocol": changed_split}, expected)

    def test_legacy_npz_still_uses_legacy_adapter(self):
        lists = self.root / "lists"
        lists.mkdir()
        (lists / "train.txt").write_text("case0001_slice000\n", encoding="utf-8")
        for modality in ("bSSFP", "LGE", "T2w"):
            folder = self.root / modality / "train_npz"
            folder.mkdir(parents=True)
            np.savez(folder / "case0001_slice000.npz", image=np.zeros((2, 2), np.float32), label=np.array([[0, 1], [2, 3]], np.uint8))
        sample = MyopsDataset(self.root, lists)[0]
        np.testing.assert_array_equal(sample["label"], [[0, 1], [3, 2]])
        self.assertEqual(sample["image"].shape, (3, 2, 2))

    def test_both_formats_have_identical_training_augmentation(self):
        lists = self.root / "lists"
        lists.mkdir()
        (lists / "train.txt").write_text("equivalent_slice000\n", encoding="utf-8")
        legacy = np.zeros((128, 128), dtype=np.uint8)
        legacy[self.raw[:, :, 1] == 200] = 1
        legacy[self.raw[:, :, 1] == 2221] = 2
        legacy[self.raw[:, :, 1] == 1220] = 3
        for channel, modality in enumerate(("bSSFP", "LGE", "T2w")):
            folder = self.root / modality / "train_npz"
            folder.mkdir(parents=True)
            np.savez(folder / "equivalent_slice000.npz", image=self.images[channel][:, :, 1], label=legacy)
        pp = MyopsPPDataset(self.root, transform=build_train_transform())
        original = MyopsDataset(self.root, lists, transform=build_train_transform())
        for seed in range(64):
            random.seed(seed)
            np.random.seed(seed)
            pp_sample = pp[1]
            random.seed(seed)
            np.random.seed(seed)
            legacy_sample = original[0]
            with self.subTest(seed=seed):
                self.assertTrue(torch.equal(pp_sample["image"], legacy_sample["image"]))
                self.assertTrue(torch.equal(pp_sample["label"], legacy_sample["label"]))
                self.assertEqual(pp_sample["label"].dtype, torch.long)

    def test_incompatible_resume_does_not_overwrite_run_config(self):
        run_root = self.root / "runs"
        config_path = run_root / "probe" / "logs" / "config.json"
        config_path.parent.mkdir(parents=True)
        original_config = '{"original_augmentation": "unit1"}'
        config_path.write_text(original_config, encoding="utf-8")
        checkpoint = self.root / "old_pp.pt"
        torch.save({"data_protocol": MyopsPPDataset(self.root).data_protocol,
                    "training_protocol": {"cohort": "myopspp"}}, checkpoint)
        args = SimpleNamespace(dataset="myopspp", data_root=str(self.root), list_dir="unused",
                               run_root=str(run_root), run_name="probe", seed=1234, batch_size=1,
                               lr=0.0003333, weight_decay=0.0001, epochs=1, amp="none", device="cpu",
                               num_workers=0, resume=str(checkpoint), pretrained=None)
        with patch("train_myops.parse_args", return_value=args), \
             patch("train_myops.VisionTransformer", side_effect=lambda *a, **k: torch.nn.Conv2d(3, 4, 1)), \
             patch("train_myops.logging.basicConfig"), \
             patch("train_myops.logging.FileHandler", return_value=logging.NullHandler()):
            with self.assertRaisesRegex(ValueError, "augmentation differs"):
                train_myops.main()
        self.assertEqual(config_path.read_text(encoding="utf-8"), original_config)


if __name__ == "__main__":
    unittest.main()
