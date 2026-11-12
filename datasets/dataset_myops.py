"""MyoPS380 dataset and I-MMSeg augmentation for TransUNet.

Converts legacy disk format (0:bg, 1:normal, 2:scar, 3:edema) to canonical format
(0:bg, 1:normal, 2:edema, 3:scar) via canonicalize_label() target adapter.
Early-fuses the 3 modalities [bSSFP, LGE, T2w] into a (3, H, W) tensor for TransUNet.
"""
from __future__ import annotations

import os
from pathlib import Path
import random
import re
from typing import Any, Sequence

import h5py
import numpy as np
from scipy import ndimage
from scipy.ndimage import zoom
import torch
from torch.utils.data import Dataset


def canonicalize_label(label: np.ndarray, order: str = "legacy") -> np.ndarray:
    """Convert label from legacy [0:bg, 1:normal, 2:scar, 3:edema]
    to canonical [0:bg, 1:normal, 2:edema, 3:scar] using LUT [0, 1, 3, 2].
    """
    label = np.asarray(label)
    if not np.isfinite(label).all() or not np.equal(label, np.round(label)).all():
        raise ValueError("Segmentation masks must contain finite integer class IDs.")
    if label.size == 0 or label.min() < 0 or label.max() > 3:
        raise ValueError(f"Expected class IDs 0..3; found {np.unique(label).tolist()}.")
    result = label.astype(np.uint8, copy=True)
    if order in ("legacy", "background_normal_scar_edema"):
        lut = np.asarray([0, 1, 3, 2], dtype=np.uint8)
        result = lut[result]
    return result


def random_rot_flip(image, image1, image2, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    image1 = np.rot90(image1, k)
    image2 = np.rot90(image2, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    image1 = np.flip(image1, axis=axis).copy()
    image2 = np.flip(image2, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, image1, image2, label


def random_rotate(image, image1, image2, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    image1 = ndimage.rotate(image1, angle, order=0, reshape=False)
    image2 = ndimage.rotate(image2, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, image1, image2, label


def add_gaussian_noise(image, image1, image2, label, mean=0, std=0.2):
    noise = np.random.normal(mean, std, image.shape)
    image = image + noise
    image1 = image1 + noise
    image2 = image2 + noise
    image = np.clip(image, 0, 1)
    image1 = np.clip(image1, 0, 1)
    image2 = np.clip(image2, 0, 1)
    return image, image1, image2, label


class RandomGenerator:
    """Reproduces I-MMSeg augmentation (6 steps, synced RNG across modalities)."""

    def __init__(self, output_size: Sequence[int] = (128, 128), intensity_max: float = 255.0):
        self.output_size = tuple(output_size)
        # Preserve the original MyoPS380 behavior, shared by both training datasets.
        if not np.isfinite(intensity_max) or intensity_max <= 0:
            raise ValueError("intensity_max must be finite and positive")
        self.intensity_max = float(intensity_max)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        image, image1, image2, label = sample["image"], sample["image1"], sample["image2"], sample["label"]

        # Step 1: Rotation / Flip / Noise
        if random.random() > 0.5:
            image, image1, image2, label = random_rot_flip(image, image1, image2, label)
        elif random.random() > 0.5:
            image, image1, image2, label = random_rotate(image, image1, image2, label)
        elif random.random() > 0.5:
            image, image1, image2, label = add_gaussian_noise(image, image1, image2, label)

        # Step 2: Translation shift
        if random.random() > 0.5:
            shift_x = random.randint(-5, 5)
            shift_y = random.randint(-5, 5)
            image = np.roll(image, (shift_x, shift_y), axis=(0, 1))
            image1 = np.roll(image1, (shift_x, shift_y), axis=(0, 1))
            image2 = np.roll(image2, (shift_x, shift_y), axis=(0, 1))
            label = np.roll(label, (shift_x, shift_y), axis=(0, 1))

        # Step 3: Brightness
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, self.intensity_max)
            image1 = np.clip(image1 * factor, 0, self.intensity_max)
            image2 = np.clip(image2 * factor, 0, self.intensity_max)

        # Step 4: Gamma
        if random.random() > 0.5:
            gamma = random.uniform(0.8, 1.2)
            image = np.power(image / self.intensity_max, gamma) * self.intensity_max
            image1 = np.power(image1 / self.intensity_max, gamma) * self.intensity_max
            image2 = np.power(image2 / self.intensity_max, gamma) * self.intensity_max

        # Step 5: Random crop
        if random.random() > 0.5:
            top = random.randint(0, image.shape[0] // 5)
            left = random.randint(0, image.shape[1] // 5)
            image = image[top:top + self.output_size[0], left:left + self.output_size[1]]
            image1 = image1[top:top + self.output_size[0], left:left + self.output_size[1]]
            image2 = image2[top:top + self.output_size[0], left:left + self.output_size[1]]
            label = label[top:top + self.output_size[0], left:left + self.output_size[1]]

        # Step 6: Zoom / resize if necessary
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            image1 = zoom(image1, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            image2 = zoom(image2, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        if self.intensity_max == 1.0:
            # Cubic interpolation can overshoot the normalized intensity range.
            image, image1, image2 = (np.clip(a, 0, 1) for a in (image, image1, image2))

        # Early-fusion: stack (3, H, W)
        stacked = np.stack([image.astype(np.float32), image1.astype(np.float32), image2.astype(np.float32)], axis=0)
        image_tensor = torch.from_numpy(stacked).float()
        label_tensor = torch.from_numpy(label.astype(np.int64)).long()

        return {
            "image": image_tensor,
            "label": label_tensor,
            "case_name": sample.get("case_name", ""),
        }


def read_split_lines(list_dir: str | Path, split: str) -> list[str]:
    list_path = Path(list_dir) / f"{split}.txt"
    if not list_path.is_file():
        raise FileNotFoundError(f"Missing split manifest: {list_path}")
    lines = [line.strip() for line in list_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    return lines


class MyopsDataset(Dataset):
    """2D Slice Dataset for MyoPS380 training.
    
    Reads 3 aligned modalities (bSSFP, LGE, T2w) from train_npz/ folders,
    converts legacy target labels to canonical labels, and outputs
    early-fusion (3, 128, 128) tensors.
    """

    def __init__(
        self,
        data_root: str | Path,
        list_dir: str | Path,
        split: str = "train",
        transform: Any = None,
        label_order: str = "legacy",
    ):
        self.data_root = Path(data_root)
        self.list_dir = Path(list_dir)
        self.split = split
        self.transform = transform
        self.label_order = label_order
        self.sample_list = read_split_lines(self.list_dir, split)

        self.modality_roots = [
            self.data_root / "bSSFP",
            self.data_root / "LGE",
            self.data_root / "T2w",
        ]

    def __len__(self) -> int:
        return len(self.sample_list)

    def _read_slice(self, root: Path, sample_name: str) -> tuple[np.ndarray, np.ndarray | None]:
        file_path = root / "train_npz" / f"{sample_name}.npz"
        if not file_path.is_file():
            # Fallback to test_npz or direct path
            alt = root / "test_npz" / f"{sample_name}.npz"
            file_path = alt if alt.is_file() else file_path
        if not file_path.is_file():
            raise FileNotFoundError(f"Could not find slice file: {file_path}")

        with np.load(file_path, allow_pickle=False) as data:
            image = data["image"].astype(np.float32)
            label = data["label"] if "label" in data else None
        return image, label

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample_name = self.sample_list[idx]

        # Read bSSFP, LGE, T2w
        img_bssfp, label = self._read_slice(self.modality_roots[0], sample_name)
        img_lge, _ = self._read_slice(self.modality_roots[1], sample_name)
        img_t2w, _ = self._read_slice(self.modality_roots[2], sample_name)

        if label is None:
            raise ValueError(f"Target label missing in {sample_name}")

        # Canonicalize label from legacy to canonical
        label = canonicalize_label(label, order=self.label_order)

        sample = {
            "image": img_bssfp,
            "image1": img_lge,
            "image2": img_t2w,
            "label": label,
            "case_name": sample_name,
        }

        if self.transform is not None:
            sample = self.transform(sample)
        else:
            stacked = np.stack([img_bssfp, img_lge, img_t2w], axis=0)
            sample = {
                "image": torch.from_numpy(stacked).float(),
                "label": torch.from_numpy(label.astype(np.int64)).long(),
                "case_name": sample_name,
            }

        return sample


class MyopsVolumeDataset(Dataset):
    """3D Volume Dataset for MyoPS380 testing / evaluation.
    
    Reads 3D volumes (H, W, D) for official 76 test cases from test_vol_h5/ folders.
    Canonicalizes the ground truth masks if present.
    """

    def __init__(
        self,
        data_root: str | Path,
        list_dir: str | Path,
        split: str = "test_vol",
        label_order: str = "legacy",
    ):
        self.data_root = Path(data_root)
        self.list_dir = Path(list_dir)
        self.split = split
        self.label_order = label_order
        self.sample_list = read_split_lines(self.list_dir, split)

        self.modality_roots = [
            self.data_root / "bSSFP",
            self.data_root / "LGE",
            self.data_root / "T2w",
        ]

    def __len__(self) -> int:
        return len(self.sample_list)

    def _read_volume(self, root: Path, case_name: str) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
        h5_path = root / "test_vol_h5" / f"{case_name}.npy.h5"
        if h5_path.is_file():
            with h5py.File(h5_path, "r") as f:
                image = f["image"][:]
                label = f["label"][:] if "label" in f else None
                metadata = dict(f.attrs)
                for key in ("spacing", "affine"):
                    if key in f:
                        metadata[key] = f[key][:]
            return np.asarray(image, dtype=np.float32), label, metadata

        # Fallback: Reassemble slices if .npy.h5 is not present
        slice_dir = root / "train_npz"
        if not slice_dir.exists():
            slice_dir = root / "test_npz"
        paths = list(slice_dir.glob(f"{case_name}_slice*.npz"))
        indexed = []
        for path in paths:
            match = re.fullmatch(re.escape(case_name) + r"_slice(\d+)", path.stem)
            if match:
                indexed.append((int(match[1]), path))
        indexed.sort()
        if not indexed:
            raise FileNotFoundError(f"Neither test_vol_h5 nor slice files found for {case_name} in {root}")

        slices = []
        labels = []
        metadata = {}
        for _, p in indexed:
            with np.load(p, allow_pickle=False) as d:
                slices.append(d["image"])
                if "label" in d:
                    labels.append(d["label"])
        image = np.stack(slices, axis=-1).astype(np.float32)
        label = np.stack(labels, axis=-1) if labels else None
        return image, label, metadata

    def __getitem__(self, idx: int) -> dict[str, Any]:
        case_name = self.sample_list[idx]

        img_bssfp, label, meta = self._read_volume(self.modality_roots[0], case_name)
        img_lge, _, _ = self._read_volume(self.modality_roots[1], case_name)
        img_t2w, _, _ = self._read_volume(self.modality_roots[2], case_name)

        if label is not None:
            label = canonicalize_label(label, order=self.label_order)

        return {
            "image": img_bssfp,
            "image1": img_lge,
            "image2": img_t2w,
            "label": label,
            "case_name": case_name,
            "spacing": meta.get("spacing", None),
            "affine": meta.get("affine", None),
        }
