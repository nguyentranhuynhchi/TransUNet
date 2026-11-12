"""Read the aligned, normalized MyoPS++ ROI128 export without changing TransUNet.

NIfTI XYZ arrays are exposed as HWD volumes; 2D inputs use [C0, LGE, T2].
Raw masks are mapped ONCE to the canonical classes shared by CE, Dice and metrics.
This loader deliberately never calls the MyoPS380 legacy label adapter.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


CANONICAL_CLASSES = {0: "background", 1: "normal_myocardium", 2: "edema", 3: "scar"}
RAW_TO_CANONICAL = {0: 0, 200: 1, 500: 0, 600: 0, 1220: 2, 2221: 3}
MODALITIES = ("C0", "LGE", "T2")


def map_myopspp_label(label: np.ndarray) -> np.ndarray:
    """Map raw NIfTI IDs to 0:bg, 1:normal, 2:edema, 3:scar; reject other IDs."""
    label = np.asarray(label)
    if not label.size or not np.isfinite(label).all():
        raise ValueError("MyoPS++ masks must be nonempty and finite.")
    if not np.isin(label, list(RAW_TO_CANONICAL)).all():
        raise ValueError(f"Expected raw MyoPS++ label IDs {list(RAW_TO_CANONICAL)}; found {np.unique(label).tolist()}")
    result = np.zeros(label.shape, dtype=np.uint8)
    for source_id, target_id in RAW_TO_CANONICAL.items():
        result[label == source_id] = target_id
    return result


def resolve_dataset_format(data_root: str | Path, requested: str = "auto") -> str:
    if requested not in ("auto", "myops380", "myopspp"):
        raise ValueError(f"Unsupported dataset: {requested}")
    if requested != "auto":
        return requested
    root = Path(data_root)
    if (root / "splits.json").is_file() and (root / "cases").is_dir():
        return "myopspp"
    if all((root / name).is_dir() for name in ("bSSFP", "LGE", "T2w")):
        return "myops380"
    raise FileNotFoundError(f"Cannot detect dataset at {root}: expected cases/ + splits.json, or bSSFP/LGE/T2w folders.")


def validate_data_checkpoint(checkpoint: dict[str, Any], expected: dict[str, Any] | None) -> None:
    """Do not silently resume/evaluate with a different cohort, split or class order.

    Old MyoPS380 checkpoints have no data_protocol and remain supported for MyoPS380.
    MyoPS++ checkpoints must explicitly describe their canonical target adapter.
    """
    if checkpoint.get("data_protocol") != expected:
        raise ValueError("Checkpoint dataset/split/label mapping differs from the requested dataset. Use the matching checkpoint or a new run name.")


class _MyopsPPBase(Dataset):
    def __init__(self, data_root: str | Path, split: str):
        self.data_root = Path(data_root)
        split_path = self.data_root / "splits.json"
        manifest = json.loads(split_path.read_text(encoding="utf-8-sig"))
        self.patients = manifest["patients"]
        if not {"train", "test_vol"}.issubset(self.patients):
            raise ValueError("MyoPS++ splits.json must define train and test_vol patients.")
        seen: set[str] = set()
        for name, ids in self.patients.items():
            if not isinstance(ids, list) or not ids or any(not isinstance(cid, str) or not cid for cid in ids):
                raise ValueError(f"Invalid or empty patient split: {name}")
            if len(ids) != len(set(ids)) or seen.intersection(ids):
                raise ValueError(f"Duplicate patients or overlapping splits: {name}")
            if manifest.get("patient_counts", {}).get(name, len(ids)) != len(ids):
                raise ValueError(f"Patient count does not match split {name}")
            seen.update(ids)
        if split not in self.patients:
            raise ValueError(f"Unknown MyoPS++ split: {split}")
        self.case_ids = list(self.patients[split])
        self.split = split
        self.case_dirs: dict[str, Path] = {}
        for path in sorted((self.data_root / "cases").glob("*/*")):
            if path.is_dir() and path.name in seen:
                if path.name in self.case_dirs:
                    raise ValueError(f"Duplicate case directory: {path.name}")
                self.case_dirs[path.name] = path
        if seen.difference(self.case_dirs):
            raise FileNotFoundError(f"Missing case directories: {sorted(seen.difference(self.case_dirs))}")

        self.shapes: dict[str, tuple[int, ...]] = {}
        self.geometry: dict[str, dict[str, Any]] = {}
        for case in self.case_ids:
            reference = None
            for tag in (*MODALITIES, "gd"):
                path = self.case_dirs[case] / f"{case}_{tag}.nii.gz"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing MyoPS++ modality/mask: {path}")
                reader = sitk.ImageFileReader()
                reader.SetFileName(str(path))
                reader.ReadImageInformation()
                if reader.GetDimension() != 3 or reader.GetSize()[:2] != (128, 128) or reader.GetSize()[2] < 1:
                    raise ValueError(f"Expected a 128x128xD ROI volume: {path}")
                geometry = (reader.GetSize(), reader.GetSpacing(), reader.GetOrigin(), reader.GetDirection())
                if reference is None:
                    reference = geometry
                    self.shapes[case] = geometry[0]
                    spacing, origin, direction = geometry[1:]
                    affine = np.eye(4)
                    affine[:3, :3] = np.asarray(direction).reshape(3, 3) @ np.diag(spacing)
                    affine[:3, 3] = origin
                    # SimpleITK uses LPS physical coordinates; NIfTI affines use RAS.
                    affine = np.diag([-1, -1, 1, 1]) @ affine
                    self.geometry[case] = {"spacing": np.asarray(spacing), "affine": affine}
                elif any(not np.allclose(a, b, rtol=0, atol=1e-5) for a, b in zip(reference, geometry)):
                    raise ValueError(f"Unaligned MyoPS++ modalities/mask: {path}")

        self.data_protocol = {
            "dataset": "myopspp",
            "dataset_id": manifest["dataset_id"],
            "split_protocol": manifest["split_protocol"],
            "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            "input_modalities": list(MODALITIES),
            "axis_order": "HWD=NIfTI_XYZ",
            "image_range": [0.0, 1.0],
            "source_labels": "raw_myopspp",
            "raw_to_canonical": {str(k): v for k, v in RAW_TO_CANONICAL.items()},
            "canonical_classes": {str(k): v for k, v in CANONICAL_CLASSES.items()},
        }
        # Bounded, per-worker volume cache; augmentation must not modify these arrays.
        self._cache: OrderedDict = OrderedDict()

    def _read_case(self, case: str) -> tuple[np.ndarray, ...]:
        if case not in self._cache:
            arrays = []
            for tag in (*MODALITIES, "gd"):
                image = sitk.ReadImage(str(self.case_dirs[case] / f"{case}_{tag}.nii.gz"))
                array = sitk.GetArrayFromImage(image).transpose(2, 1, 0)  # ZYX -> XYZ (HWD)
                if tag == "gd":
                    array = map_myopspp_label(array)
                else:
                    array = array.astype(np.float32)
                    if not np.isfinite(array).all() or array.min() < -1e-6 or array.max() > 1 + 1e-6:
                        raise ValueError(f"Expected normalized finite [0,1] images: {case}_{tag}")
                    array = np.clip(array, 0, 1)
                arrays.append(np.ascontiguousarray(array))
            self._cache[case] = tuple(arrays)
            if len(self._cache) > 8:
                self._cache.popitem(last=False)
        self._cache.move_to_end(case)
        return self._cache[case]


class MyopsPPDataset(_MyopsPPBase):
    """Train only on slices from the patients listed in splits.json (no re-split)."""
    def __init__(self, data_root: str | Path, transform: Any = None):
        super().__init__(data_root, "train")
        self.transform = transform
        self.slice_index = [(case, z) for case in self.case_ids for z in range(self.shapes[case][2])]
        self.sample_list = [f"{case}_slice{z:03d}" for case, z in self.slice_index]

    def __len__(self) -> int:
        return len(self.slice_index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        case, z = self.slice_index[idx]
        images = self._read_case(case)
        sample = dict(zip(("image", "image1", "image2", "label"), (a[:, :, z].copy() for a in images)))
        sample["case_name"] = self.sample_list[idx]
        if self.transform is not None:
            return self.transform(sample)
        return {
            "image": torch.from_numpy(np.stack([sample[k] for k in ("image", "image1", "image2")])),
            "label": torch.from_numpy(sample["label"].astype(np.int64)),
            "case_name": sample["case_name"],
        }


class MyopsPPVolumeDataset(_MyopsPPBase):
    def __init__(self, data_root: str | Path):
        super().__init__(data_root, "test_vol")

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        case = self.case_ids[idx]
        images = self._read_case(case)
        return {
            **dict(zip(("image", "image1", "image2", "label"), images)),
            "case_name": case,
            **self.geometry[case],
        }
