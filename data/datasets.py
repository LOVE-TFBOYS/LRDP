from pathlib import Path

from torch.utils.data import Dataset

from .image_io import get_case_id, load_image, to_tensor
from .pairing import build_atlas_to_subject_pairs, build_subject_to_subject_pairs, split_pairs


def normalize_oasis_split(split):
    split = str(split or "train").lower()
    if split in {"train", "tr", "imagestr"}:
        return "train"
    if split in {"test", "ts", "imagests"}:
        return "test"
    raise ValueError(f"Unknown data.split={split}. Expected one of train/tr/imagesTr/test/ts/imagesTs.")


def resolve_oasis_dirs(config=None, data_cfg=None, split=None):
    if data_cfg is None:
        data_cfg = (config or {}).get("data", {}) if isinstance(config, dict) else {}
    data_root = Path(data_cfg.get("root", data_cfg.get("data_root", "/data/TFBOYS/dataset/OASIS")))
    split_name = normalize_oasis_split(split if split is not None else data_cfg.get("split", "train"))

    if split_name == "train":
        default_image_dir = "imagesTr"
        default_label_dir = "labelsTr"
        default_mask_dir = "masksTr"
    else:
        default_image_dir = "imagesTs"
        default_label_dir = None
        default_mask_dir = "masksTs"

    image_dir = data_cfg["image_dir"] if "image_dir" in data_cfg else default_image_dir
    label_dir = data_cfg["label_dir"] if "label_dir" in data_cfg else default_label_dir
    mask_dir = data_cfg["mask_dir"] if "mask_dir" in data_cfg else default_mask_dir
    image_glob = data_cfg.get("image_glob", "*.nii.gz")

    return {
        "data_root": data_root,
        "split": split_name,
        "image_root": data_root / image_dir,
        "label_root": data_root / label_dir if label_dir else None,
        "mask_root": data_root / mask_dir if mask_dir else None,
        "image_dir": image_dir,
        "label_dir": label_dir,
        "mask_dir": mask_dir,
        "image_glob": image_glob,
    }


class BaseRegistrationDataset(Dataset):
    def __init__(self, pairs, preprocess=None, transform=None):
        self.pairs = list(pairs)
        self.preprocess = preprocess
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def load_case(self, path, seg_path=None, mask_path=None):
        image, meta = load_image(path)
        seg, seg_meta = (None, None)
        mask, mask_meta = (None, None)
        if seg_path is not None:
            seg, seg_meta = load_image(seg_path)
        if mask_path is not None:
            mask, mask_meta = load_image(mask_path)
        if self.preprocess is not None:
            image, meta, seg, mask = self.preprocess(image, meta, seg=seg, mask=mask)
        return image, meta, seg, mask, seg_meta, mask_meta


class PairRegistrationDataset(BaseRegistrationDataset):
    def __getitem__(self, index):
        pair = self.pairs[index]
        fixed, fixed_meta, fixed_seg, fixed_mask, _, _ = self.load_case(pair["fixed"], pair.get("fixed_seg"), pair.get("fixed_mask"))
        moving, moving_meta, moving_seg, moving_mask, _, _ = self.load_case(pair["moving"], pair.get("moving_seg"), pair.get("moving_mask"))
        sample = {
            "fixed": to_tensor(fixed),
            "moving": to_tensor(moving),
            "fixed_path": str(pair["fixed"]),
            "moving_path": str(pair["moving"]),
            "fixed_id": fixed_meta.get("case_id", get_case_id(pair["fixed"])),
            "moving_id": moving_meta.get("case_id", get_case_id(pair["moving"])),
            "fixed_meta": fixed_meta,
            "moving_meta": moving_meta,
        }
        if fixed_seg is not None:
            sample["fixed_seg"] = to_tensor(fixed_seg)
        if moving_seg is not None:
            sample["moving_seg"] = to_tensor(moving_seg)
        if fixed_mask is not None:
            sample["fixed_mask"] = to_tensor(fixed_mask)
        if moving_mask is not None:
            sample["moving_mask"] = to_tensor(moving_mask)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class OASISRegistrationDataset(PairRegistrationDataset):
    def __init__(
        self,
        data_root=None,
        image_dir=None,
        label_dir=None,
        mask_dir=None,
        image_glob="*.nii.gz",
        label_suffix="_seg",
        mask_suffix="_mask",
        pair_strategy="atlas_to_subject",
        atlas_path=None,
        split="train",
        split_ratio=(0.7, 0.1, 0.2),
        seed=42,
        preprocess=None,
        transform=None,
        num_pairs=None,
        oasis_split=None,
    ):
        directory_split = oasis_split or ("test" if split in {"test", "ts", "imagesTs"} else "train")
        data_cfg = {"root": data_root, "image_glob": image_glob, "split": directory_split}
        if image_dir is not None:
            data_cfg["image_dir"] = image_dir
        if label_dir is not None:
            data_cfg["label_dir"] = label_dir
        if mask_dir is not None:
            data_cfg["mask_dir"] = mask_dir
        paths = resolve_oasis_dirs(data_cfg=data_cfg)
        data_root = paths["data_root"]
        image_paths = sorted(paths["image_root"].glob(paths["image_glob"]))
        if not image_paths:
            raise FileNotFoundError(f"No images found in {paths['image_root']} with glob {paths['image_glob']}")

        if paths["label_root"] is not None and not paths["label_root"].exists():
            print(f"[Warning] label directory not found: {paths['label_root']}")
        if paths["mask_root"] is not None and not paths["mask_root"].exists():
            print(f"[Warning] mask directory not found: {paths['mask_root']}")

        if pair_strategy == "atlas_to_subject":
            atlas_path = Path(atlas_path) if atlas_path is not None else image_paths[0]
            pairs = build_atlas_to_subject_pairs(image_paths, atlas_path)
        elif pair_strategy == "subject_to_subject":
            pairs = build_subject_to_subject_pairs(image_paths, mode="random", num_pairs=num_pairs, seed=seed)
        elif pair_strategy == "all":
            pairs = build_subject_to_subject_pairs(image_paths, mode="all", seed=seed)
        else:
            raise ValueError("pair_strategy must be atlas_to_subject, subject_to_subject, or all")

        if paths["label_root"] is not None or paths["mask_root"] is not None:
            for pair in pairs:
                for role in ("fixed", "moving"):
                    image_path = Path(pair[role])
                    case_id = get_case_id(image_path)
                    if paths["label_root"] is not None:
                        pair[f"{role}_seg"] = paths["label_root"] / f"{case_id}{label_suffix}.nii.gz"
                        if not Path(pair[f"{role}_seg"]).exists():
                            pair.pop(f"{role}_seg")
                    if paths["mask_root"] is not None:
                        pair[f"{role}_mask"] = paths["mask_root"] / f"{case_id}{mask_suffix}.nii.gz"
                        if not Path(pair[f"{role}_mask"]).exists():
                            pair.pop(f"{role}_mask")

        if paths["split"] == "test":
            selected = pairs
        else:
            splits = split_pairs(pairs, split_ratio=split_ratio, seed=seed)
            selected = splits.get(split, pairs)
        super().__init__(selected, preprocess=preprocess, transform=transform)
