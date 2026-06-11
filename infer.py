import argparse
import csv
import inspect
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import OASISRegistrationDataset, PreprocessPipeline, save_image
from models import LRDPRegistrationModel
from utils import config_section, load_config, set_by_path


def parse_args():
    parser = argparse.ArgumentParser(description="Batch LRDP inference from an experiment config.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None, help="Optional inference config. Defaults to checkpoint config.")
    parser.add_argument("--data-root", default=None, help="Override data.root.")
    parser.add_argument("--output-dir", default=None, help="Override inference.output_dir.")
    parser.add_argument("--split", default=None, choices=["train", "val", "test"], help="Override inference.split.")
    parser.add_argument("--device", default=None, help="Override inference.device. Use cpu/cuda/cuda:0.")
    parser.add_argument("--save-nifti", action="store_true", help="Override inference.save_nifti to true.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override any config key, e.g. --set inference.batch_size=2.")
    return parser.parse_args()


def _filter_kwargs(callable_obj, values):
    signature = inspect.signature(callable_obj)
    allowed = {name for name in signature.parameters if name != "self"}
    return {key: value for key, value in values.items() if key in allowed}


def resolve_device(value):
    if value in {None, "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def apply_named_overrides(config, args):
    if args.data_root is not None:
        set_by_path(config, "data.root", args.data_root)
    if args.output_dir is not None:
        set_by_path(config, "inference.output_dir", args.output_dir)
    if args.split is not None:
        set_by_path(config, "inference.split", args.split)
    if args.device is not None:
        set_by_path(config, "inference.device", args.device)
    if args.save_nifti:
        set_by_path(config, "inference.save_nifti", "true")
    return config


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
        save_processed=preprocessing.get("save_processed", False),
        processed_dir=preprocessing.get("processed_dir"),
        overwrite_processed=preprocessing.get("overwrite_processed", False),
        save_processed_metadata=preprocessing.get("save_processed_metadata", False),
    )


def build_dataset(config):
    data = config_section(config, "data")
    inference = config_section(config, "inference")
    return OASISRegistrationDataset(
        data_root=data.get("root", data.get("data_root")),
        image_dir=data.get("image_dir"),
        label_dir=data.get("label_dir"),
        mask_dir=data.get("mask_dir"),
        image_glob=data.get("image_glob", "*.nii.gz"),
        label_suffix=data.get("label_suffix", "_seg"),
        mask_suffix=data.get("mask_suffix", "_mask"),
        pair_strategy=data.get("pair_strategy", "atlas_to_subject"),
        atlas_path=data.get("atlas_path"),
        split=inference.get("split", "test"),
        split_ratio=data.get("split_ratio", (0.7, 0.1, 0.2)),
        seed=int(data.get("seed", 42)),
        preprocess=build_preprocess(config),
        transform=None,
        num_pairs=data.get("num_pairs"),
        oasis_split=inference.get("split", data.get("split", "test")),
    )


def load_inference_config(args, checkpoint):
    if args.config is not None:
        config = load_config(args.config, overrides=args.set)
    else:
        config = checkpoint.get("config") or load_config("configs/lrdp_default.yaml", overrides=args.set)
        for override in args.set:
            key, sep, value = override.partition("=")
            if not sep:
                raise ValueError(f"Override must use key=value syntax, got: {override}")
            set_by_path(config, key.strip(), value.strip())
    return apply_named_overrides(config, args)


def build_model(checkpoint, config, device):
    model_config = checkpoint.get("model_config")
    if model_config is None:
        model_config = dict(config_section(config, "model"))
        model_config.pop("name", None)
    model = LRDPRegistrationModel(**_filter_kwargs(LRDPRegistrationModel, model_config)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def move_batch(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def batch_item(value, item):
    if torch.is_tensor(value):
        return value[item]
    if isinstance(value, (list, tuple)):
        return value[item]
    return value


def meta_item(meta, item):
    if not isinstance(meta, dict):
        return meta
    return {key: batch_item(value, item) for key, value in meta.items()}


def tensor_to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return value


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = load_inference_config(args, checkpoint)
    inference = config_section(config, "inference")
    device = torch.device(resolve_device(inference.get("device", "auto")))

    model = build_model(checkpoint, config, device)
    output_dir = Path(inference.get("output_dir", "inference_outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_inference_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    loader = DataLoader(
        build_dataset(config),
        batch_size=int(inference.get("batch_size", config_section(config, "dataloader").get("batch_size", 1))),
        shuffle=False,
        num_workers=int(inference.get("num_workers", config_section(config, "dataloader").get("num_workers", 2))),
        pin_memory=bool(config_section(config, "dataloader").get("pin_memory", True)) and device.type == "cuda",
    )
    manifest_rows = []

    with torch.no_grad():
        for index, batch in enumerate(loader):
            batch = move_batch(batch, device)
            outputs = model(batch["fixed"], batch["moving"], return_intermediates=True)
            for item in range(outputs["warped"].shape[0]):
                fixed_id = batch_item(batch["fixed_id"], item)
                moving_id = batch_item(batch["moving_id"], item)
                case_dir = output_dir / f"{index:04d}_{moving_id}_to_{fixed_id}"
                case_dir.mkdir(parents=True, exist_ok=True)

                fixed_meta = meta_item(batch.get("fixed_meta"), item)
                moving_meta = meta_item(batch.get("moving_meta"), item)
                payload = {
                    "warped": outputs["warped"][item].cpu(),
                    "flow": outputs["flow"][item].cpu(),
                    "fixed_id": fixed_id,
                    "moving_id": moving_id,
                    "fixed_meta": fixed_meta,
                    "moving_meta": moving_meta,
                }
                torch.save(payload, case_dir / "result.pt")

                if bool(inference.get("save_nifti", False)):
                    affine = tensor_to_numpy(fixed_meta.get("affine")) if isinstance(fixed_meta, dict) else None
                    save_image(outputs["warped"][item, 0].cpu().numpy(), case_dir / "warped.nii.gz", affine=affine)
                    save_image(outputs["flow"][item].permute(1, 2, 3, 0).cpu().numpy(), case_dir / "flow.npy")

                result_path = case_dir / "result.pt"
                manifest_rows.append({"moving_id": moving_id, "fixed_id": fixed_id, "path": str(result_path)})
                print(f"saved {result_path}")

    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["moving_id", "fixed_id", "path"])
        writer.writeheader()
        writer.writerows(manifest_rows)


if __name__ == "__main__":
    main()
