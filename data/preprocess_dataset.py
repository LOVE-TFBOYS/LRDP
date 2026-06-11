import argparse
import csv
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.datasets import resolve_oasis_dirs
from data.image_io import get_case_id, load_image
from data.preprocess import PreprocessPipeline
from utils import config_section, load_config, set_by_path


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess and save LRDP registration volumes.")
    parser.add_argument("--config", default="configs/lrdp_default.yaml")
    parser.add_argument("--data-root", default=None, help="Override data.root.")
    parser.add_argument("--processed-dir", default=None, help="Override data.output_dir.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing processed files.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override any config key.")
    return parser.parse_args()


def apply_overrides(config, args):
    if args.data_root is not None:
        set_by_path(config, "data.root", args.data_root)
    if args.processed_dir is not None:
        set_by_path(config, "data.output_dir", args.processed_dir)
    if args.overwrite:
        set_by_path(config, "preprocess.overwrite_processed", "true")
    set_by_path(config, "preprocess.save_processed", "true")
    return config


def find_optional_case_file(directory, image_path, case_id, suffix):
    if directory is None or not Path(directory).exists():
        return None
    path = Path(directory) / f"{case_id}{suffix}.nii.gz"
    if path.exists():
        return path
    same_name_path = Path(directory) / image_path.name
    if same_name_path.exists():
        return same_name_path
    npy_path = Path(directory) / f"{case_id}{suffix}.npy"
    if npy_path.exists():
        return npy_path
    return None


def get_preprocess_config(config):
    preprocessing = dict(config_section(config, "preprocessing"))
    preprocessing.update(config_section(config, "preprocess"))
    data = config_section(config, "data")
    if "processed_dir" not in preprocessing and data.get("output_dir") is not None:
        preprocessing["processed_dir"] = data.get("output_dir")
    if "normalize_method" not in preprocessing and "normalize" in preprocessing:
        preprocessing["normalize_method"] = "zscore" if preprocessing.get("normalize") else "none"
    return preprocessing


def build_preprocess(config):
    preprocessing = get_preprocess_config(config)
    return PreprocessPipeline(
        target_spacing=preprocessing.get("target_spacing"),
        target_shape=preprocessing.get("target_shape"),
        normalize_method=preprocessing.get("normalize_method", "zscore"),
        crop_foreground=preprocessing.get("crop_foreground", True),
        use_mask=preprocessing.get("use_mask", False),
        orientation=preprocessing.get("orientation"),
        save_processed=True,
        processed_dir=preprocessing.get("processed_dir"),
        overwrite_processed=preprocessing.get("overwrite_processed", False),
        save_processed_metadata=preprocessing.get("save_processed_metadata", False),
    )


def main():
    args = parse_args()
    config = apply_overrides(load_config(args.config, overrides=args.set), args)
    preprocessing = get_preprocess_config(config)
    paths = resolve_oasis_dirs(config)

    image_paths = sorted(paths["image_root"].glob(paths["image_glob"]))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {paths['image_root']} with glob {paths['image_glob']}")
    if paths["label_root"] is not None and not paths["label_root"].exists():
        print(f"[Warning] label directory not found: {paths['label_root']}")
    if paths["mask_root"] is not None and not paths["mask_root"].exists():
        print(f"[Warning] mask directory not found: {paths['mask_root']}")
    if not preprocessing.get("processed_dir"):
        raise ValueError("data.output_dir or preprocessing.processed_dir must be set when saving processed data")

    preprocess = build_preprocess(config)
    manifest_rows = []
    data = config_section(config, "data")
    for index, image_path in enumerate(image_paths, start=1):
        case_id = get_case_id(image_path)
        seg_path = find_optional_case_file(paths["label_root"], image_path, case_id, data.get("label_suffix", "_seg"))
        mask_path = find_optional_case_file(paths["mask_root"], image_path, case_id, data.get("mask_suffix", "_mask"))

        image, meta = load_image(image_path)
        seg = load_image(seg_path)[0] if seg_path is not None else None
        mask = load_image(mask_path)[0] if mask_path is not None else None
        _, processed_meta, _, _ = preprocess(image, meta, seg=seg, mask=mask)

        manifest_rows.append(
            {
                "case_id": case_id,
                "source_image": str(image_path),
                "source_label": str(seg_path) if seg_path is not None else "",
                "source_mask": str(mask_path) if mask_path is not None else "",
                "processed_shape": "x".join(str(v) for v in processed_meta["processed_shape"]),
            }
        )
        print(f"[{index}/{len(image_paths)}] processed {case_id}")

    processed_dir = Path(preprocessing.get("processed_dir"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    with (processed_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case_id", "source_image", "source_label", "source_mask", "processed_shape"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)


if __name__ == "__main__":
    main()
