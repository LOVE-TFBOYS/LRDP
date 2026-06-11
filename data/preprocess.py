import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .image_io import save_image
except ImportError:
    from image_io import save_image


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _save_array_if_needed(array, path, overwrite=False):
    if array is None:
        return
    path = Path(path)
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(array, path)


def _save_json_if_needed(payload, path, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def percentile_clip(image, lower=1, upper=99):
    lo, hi = np.percentile(image, (lower, upper))
    return np.clip(image, lo, hi)


def zscore_normalize(image, mask=None, eps=1e-8):
    region = image[mask > 0] if mask is not None and np.any(mask > 0) else image
    return (image - float(region.mean())) / (float(region.std()) + eps)


def minmax_normalize(image, percentiles=(1, 99), eps=1e-8):
    lo, hi = np.percentile(image, percentiles)
    image = np.clip(image, lo, hi)
    return (image - lo) / (hi - lo + eps)


def normalize_mri(image, method="zscore", mask=None):
    if method == "none" or method is None:
        return image.astype(np.float32)
    if method == "zscore":
        return zscore_normalize(percentile_clip(image), mask=mask).astype(np.float32)
    if method == "minmax":
        return minmax_normalize(image).astype(np.float32)
    raise ValueError("normalize method must be 'zscore', 'minmax', or 'none'")


def estimate_foreground_mask(image, threshold=None):
    if threshold is None:
        return np.asarray(image != 0)
    return np.asarray(image > threshold)


def compute_foreground_bbox(image, threshold=None, margin=8):
    mask = estimate_foreground_mask(image, threshold)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return tuple((0, size) for size in image.shape)
    start = np.maximum(coords.min(axis=0) - margin, 0)
    end = np.minimum(coords.max(axis=0) + margin + 1, np.asarray(image.shape))
    return tuple((int(s), int(e)) for s, e in zip(start, end))


def apply_bbox(image, bbox):
    slices = tuple(slice(start, end) for start, end in bbox)
    return image[slices]


def center_crop_or_pad(image, target_shape):
    result = image
    for axis, target in enumerate(target_shape):
        size = result.shape[axis]
        if size > target:
            start = (size - target) // 2
            result = np.take(result, indices=range(start, start + target), axis=axis)
    pad_width = []
    for size, target in zip(result.shape, target_shape):
        total = max(target - size, 0)
        before = total // 2
        pad_width.append((before, total - before))
    return np.pad(result, pad_width, mode="constant")


def crop_or_pad_to_shape(image, target_shape):
    return center_crop_or_pad(image, target_shape)


def compute_new_shape(old_shape, current_spacing, target_spacing):
    return tuple(int(round(s * cs / ts)) for s, cs, ts in zip(old_shape, current_spacing, target_spacing))


def _torch_resample(array, target_shape, mode):
    tensor = torch.as_tensor(array, dtype=torch.float32)[None, None]
    if mode == "nearest":
        out = F.interpolate(tensor, size=target_shape, mode="nearest")
    else:
        out = F.interpolate(tensor, size=target_shape, mode="trilinear", align_corners=True)
    return out[0, 0].cpu().numpy()


def resample_image(image, current_spacing, target_spacing, mode="linear"):
    target_shape = compute_new_shape(image.shape, current_spacing, target_spacing)
    return _torch_resample(image, target_shape, mode="trilinear" if mode in {"linear", "trilinear"} else mode)


def resample_label(label, current_spacing, target_spacing):
    target_shape = compute_new_shape(label.shape, current_spacing, target_spacing)
    return _torch_resample(label, target_shape, mode="nearest")


def process_segmentation(seg, target_shape=None):
    if seg is None:
        return None
    if target_shape is not None:
        seg = crop_or_pad_to_shape(seg, target_shape)
    return seg.astype(np.int64)


class PreprocessPipeline:
    def __init__(
        self,
        target_spacing=None,
        target_shape=None,
        normalize_method="zscore",
        crop_foreground=True,
        use_mask=False,
        orientation=None,
        save_processed=False,
        processed_dir=None,
        overwrite_processed=False,
        save_processed_metadata=False,
    ):
        self.target_spacing = tuple(target_spacing) if target_spacing is not None else None
        self.target_shape = tuple(target_shape) if target_shape is not None else None
        self.normalize_method = normalize_method
        self.crop_foreground = crop_foreground
        self.use_mask = use_mask
        self.orientation = orientation
        self.save_processed = save_processed
        self.processed_dir = Path(processed_dir) if processed_dir else None
        self.overwrite_processed = overwrite_processed
        self.save_processed_metadata = save_processed_metadata

    def _save_processed_case(self, image, meta, seg=None, mask=None):
        if not self.save_processed or self.processed_dir is None:
            return
        case_id = meta.get("case_id")
        if not case_id:
            raise KeyError("meta must include case_id when save_processed is enabled")
        _save_array_if_needed(image, self.processed_dir / "images" / f"{case_id}.npy", self.overwrite_processed)
        _save_array_if_needed(seg, self.processed_dir / "labels" / f"{case_id}_seg.npy", self.overwrite_processed)
        _save_array_if_needed(mask, self.processed_dir / "masks" / f"{case_id}_mask.npy", self.overwrite_processed)
        if self.save_processed_metadata:
            _save_json_if_needed(meta, self.processed_dir / "metadata" / f"{case_id}.json", self.overwrite_processed)

    def __call__(self, image, meta, seg=None, mask=None):
        processed_meta = dict(meta)
        # TODO: add robust orientation canonicalization when orientation policy is finalized.
        if self.target_spacing is not None and meta.get("spacing") is not None:
            image = resample_image(image, meta["spacing"], self.target_spacing)
            if seg is not None:
                seg = resample_label(seg, meta["spacing"], self.target_spacing)
            if mask is not None:
                mask = resample_label(mask, meta["spacing"], self.target_spacing)
            processed_meta["spacing"] = self.target_spacing

        if self.crop_foreground:
            bbox_source = mask if self.use_mask and mask is not None else image
            bbox = compute_foreground_bbox(bbox_source)
            image = apply_bbox(image, bbox)
            if seg is not None:
                seg = apply_bbox(seg, bbox)
            if mask is not None:
                mask = apply_bbox(mask, bbox)
            processed_meta["foreground_bbox"] = bbox

        if self.target_shape is not None:
            image = crop_or_pad_to_shape(image, self.target_shape)
            seg = process_segmentation(seg, self.target_shape)
            if mask is not None:
                mask = crop_or_pad_to_shape(mask, self.target_shape)

        norm_mask = mask if self.use_mask else None
        image = normalize_mri(image, self.normalize_method, mask=norm_mask)
        processed_meta["processed_shape"] = tuple(image.shape)
        self._save_processed_case(image, processed_meta, seg=seg, mask=mask)

        return image, processed_meta, seg, mask
