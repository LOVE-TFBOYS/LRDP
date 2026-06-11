from pathlib import Path

import numpy as np
import torch


def _require_nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("Reading/writing NIfTI requires nibabel. Install it with `pip install nibabel`.") from exc
    return nib


def get_case_id(path):
    path = Path(path)
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def load_nifti(path):
    nib = _require_nibabel()
    path = Path(path)
    img = nib.load(str(path))
    array = img.get_fdata(dtype=np.float32)
    spacing = np.sqrt((img.affine[:3, :3] ** 2).sum(axis=0))
    meta = {
        "affine": img.affine,
        "spacing": tuple(float(v) for v in spacing),
        "shape": tuple(array.shape),
        "path": str(path),
        "filename": path.name,
        "case_id": get_case_id(path),
    }
    return array, meta


def save_nifti(array, affine, path):
    nib = _require_nibabel()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(array), affine if affine is not None else np.eye(4)), str(path))


def load_numpy(path):
    path = Path(path)
    array = np.load(path).astype(np.float32)
    meta = {
        "affine": None,
        "spacing": None,
        "shape": tuple(array.shape),
        "path": str(path),
        "filename": path.name,
        "case_id": get_case_id(path),
    }
    return array, meta


def load_image(path):
    path = Path(path)
    name = path.name.lower()
    if name.endswith(".nii") or name.endswith(".nii.gz"):
        return load_nifti(path)
    if name.endswith(".npy"):
        return load_numpy(path)
    raise ValueError(f"Unsupported image format: {path}")


def save_image(array, path, affine=None):
    path = Path(path)
    name = path.name.lower()
    if name.endswith(".nii") or name.endswith(".nii.gz"):
        save_nifti(array, affine, path)
        return
    if name.endswith(".npy"):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.asarray(array))
        return
    raise ValueError(f"Unsupported image format: {path}")


def to_tensor(array, add_channel=True):
    tensor = torch.as_tensor(np.asarray(array), dtype=torch.float32)
    if add_channel and tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    return tensor
