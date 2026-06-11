import argparse
import inspect
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import OASISRegistrationDataset, PreprocessPipeline, RegistrationAugmentation
from losses import RegistrationLoss
from models import LRDPRegistrationModel
from models.registration import SpatialTransformer3D
from utils import batch_dice_score, config_section, folding_ratio, load_config, set_by_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train LRDP from an experiment config.")
    parser.add_argument("--config", default="configs/lrdp_default.yaml")
    parser.add_argument("--ablation", default=None, help="Name in config ablation_overrides, e.g. no_diffusion.")
    parser.add_argument("--data-root", default=None, help="Override data.root.")
    parser.add_argument("--save-dir", default=None, help="Override training.save_dir.")
    parser.add_argument("--resume", default=None, help="Override training.resume.")
    parser.add_argument("--device", default=None, help="Override training.device. Use cpu/cuda/cuda:0.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override any config key, e.g. --set training.epochs=200.")
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
    if args.save_dir is not None:
        set_by_path(config, "training.save_dir", args.save_dir)
    if args.resume is not None:
        set_by_path(config, "training.resume", args.resume)
    if args.device is not None:
        set_by_path(config, "training.device", args.device)
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


def build_dataset(config, split):
    data = config_section(config, "data")
    augmentation = config_section(config_section(config, "augmentation"), split)
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
        split=split,
        split_ratio=data.get("split_ratio", (0.7, 0.1, 0.2)),
        seed=int(data.get("seed", 42)),
        preprocess=build_preprocess(config),
        transform=RegistrationAugmentation(**augmentation),
        num_pairs=data.get("num_pairs"),
        oasis_split=data.get("split", "train"),
    )


def build_model(config, device):
    model_config = dict(config_section(config, "model"))
    model_config.pop("name", None)
    model = LRDPRegistrationModel(**_filter_kwargs(LRDPRegistrationModel, model_config))
    return model.to(device), model_config


def build_loss(config, device):
    loss_config = _filter_kwargs(RegistrationLoss, config_section(config, "loss"))
    return RegistrationLoss(**loss_config).to(device)


def build_optimizer(config, model):
    optimizer_config = dict(config_section(config, "optimizer"))
    name = optimizer_config.pop("name", "AdamW")
    if name == "AdamW":
        return torch.optim.AdamW(model.parameters(), **optimizer_config)
    if name == "Adam":
        return torch.optim.Adam(model.parameters(), **optimizer_config)
    raise ValueError(f"Unsupported optimizer: {name}")


def move_batch(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def current_lr(optimizer):
    if optimizer is None:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


def format_metrics(metrics):
    text = (
        f"loss={metrics['total']:.6f}, "
        f"sim={metrics['sim']:.6f}, "
        f"smooth={metrics['smooth']:.6f}, "
        f"jac={metrics['jac']:.6f}, "
        f"diff={metrics['diff']:.6f}, "
        f"ms={metrics['multiscale']:.6f}"
    )
    if "dice_before" in metrics:
        text += f", dice_before={metrics['dice_before']:.6f}, dice_after={metrics['dice_after']:.6f}"
    if "folding_ratio" in metrics:
        text += f", fold={metrics['folding_ratio']:.6f}"
    return text


def tensor_metrics_to_float(losses):
    return {key: float(value.detach()) for key, value in losses.items()}


def run_epoch(model, loader, loss_fn, optimizer, device, epoch, train=True, print_every=1, seg_transformer=None):
    model.train(train)
    totals = {"total": 0.0, "sim": 0.0, "smooth": 0.0, "jac": 0.0, "diff": 0.0, "multiscale": 0.0}
    metric_totals = {"dice_before": 0.0, "dice_after": 0.0, "folding_ratio": 0.0}
    metric_count = {"dice": 0, "folding": 0}
    count = 0
    phase = "train" if train else "val"
    start_time = time.time()
    for step, batch in enumerate(loader, start=1):
        iter_start = time.time()
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(train):
            outputs = model(batch["fixed"], batch["moving"], return_intermediates=True)
            losses = loss_fn(outputs, batch["fixed"], batch["moving"])
            if train:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                optimizer.step()
        batch_size = int(batch["fixed"].shape[0])
        extra_metrics = {}
        if not train:
            fold = folding_ratio(outputs["flow"])
            extra_metrics["folding_ratio"] = float(fold.detach())
            metric_totals["folding_ratio"] += extra_metrics["folding_ratio"] * batch_size
            metric_count["folding"] += batch_size
            if seg_transformer is not None and "fixed_seg" in batch and "moving_seg" in batch:
                moving_seg = batch["moving_seg"].float()
                fixed_seg = batch["fixed_seg"].long()
                warped_seg = seg_transformer(moving_seg, outputs["flow"]).round().long()
                moving_seg = moving_seg.long()
                dice_before = batch_dice_score(moving_seg, fixed_seg)
                dice_after = batch_dice_score(warped_seg, fixed_seg)
                extra_metrics["dice_before"] = float(dice_before.detach())
                extra_metrics["dice_after"] = float(dice_after.detach())
                metric_totals["dice_before"] += extra_metrics["dice_before"] * batch_size
                metric_totals["dice_after"] += extra_metrics["dice_after"] * batch_size
                metric_count["dice"] += batch_size
        count += batch_size
        for key in totals:
            totals[key] += float(losses[key].detach()) * batch_size
        if print_every > 0 and (step == 1 or step % print_every == 0 or step == len(loader)):
            iter_metrics = tensor_metrics_to_float(losses)
            iter_metrics.update(extra_metrics)
            lr_text = f", lr={current_lr(optimizer):.8f}" if train else ""
            print(
                f"[{phase}] epoch={epoch} iter={step}/{len(loader)} "
                f"{format_metrics(iter_metrics)}{lr_text}, "
                f"time={time.time() - iter_start:.2f}s"
            )
    epoch_metrics = {key: value / max(count, 1) for key, value in totals.items()}
    if metric_count["dice"] > 0:
        epoch_metrics["dice_before"] = metric_totals["dice_before"] / metric_count["dice"]
        epoch_metrics["dice_after"] = metric_totals["dice_after"] / metric_count["dice"]
    if metric_count["folding"] > 0:
        epoch_metrics["folding_ratio"] = metric_totals["folding_ratio"] / metric_count["folding"]
    print(f"[{phase}] epoch={epoch} summary {format_metrics(epoch_metrics)}, time={time.time() - start_time:.2f}s")
    return epoch_metrics


def save_checkpoint(path, model, optimizer, epoch, best_val, model_config, config, command_args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val": best_val,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": model_config,
            "config": config,
            "command_args": vars(command_args),
        },
        path,
    )


def main():
    args = parse_args()
    config = load_config(args.config, ablation=args.ablation, overrides=args.set)
    config = apply_named_overrides(config, args)

    training = config_section(config, "training")
    dataloader = config_section(config, "dataloader")
    seed = int(training.get("seed", config_section(config, "data").get("seed", 42)))
    torch.manual_seed(seed)

    device_name = resolve_device(training.get("device", "auto"))
    device = torch.device(device_name)
    save_dir = Path(training.get("save_dir", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)

    pin_memory = bool(dataloader.get("pin_memory", True)) and device.type == "cuda"
    train_loader = DataLoader(
        build_dataset(config, "train"),
        batch_size=int(dataloader.get("batch_size", 1)),
        shuffle=bool(dataloader.get("shuffle_train", True)),
        num_workers=int(dataloader.get("num_workers", 4)),
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        build_dataset(config, "val"),
        batch_size=int(dataloader.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(dataloader.get("num_workers", 4)),
        pin_memory=pin_memory,
    )
    print(
        f"device={device}, save_dir={save_dir}, "
        f"train_pairs={len(train_loader.dataset)}, train_iters={len(train_loader)}, "
        f"val_pairs={len(val_loader.dataset)}, val_iters={len(val_loader)}"
    )

    model, model_config = build_model(config, device)
    optimizer = build_optimizer(config, model)
    loss_fn = build_loss(config, device)
    seg_transformer = SpatialTransformer3D(mode="nearest").to(device)
    print(f"model=LRDPRegistrationModel, params={sum(p.numel() for p in model.parameters() if p.requires_grad)}, lr={current_lr(optimizer):.8f}")

    start_epoch, best_val = 1, float("inf")
    resume = training.get("resume")
    if resume:
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val = float(checkpoint.get("best_val", best_val))

    (save_dir / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    epochs = int(training.get("epochs", 100))
    save_every = max(int(training.get("save_every", 1)), 1)
    print_every = int(training.get("print_every", 1))
    for epoch in range(start_epoch, epochs + 1):
        print(f"Training starts: epoch={epoch}/{epochs}")
        train_metrics = run_epoch(model, train_loader, loss_fn, optimizer, device, epoch=epoch, train=True, print_every=print_every)
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                loss_fn,
                optimizer,
                device,
                epoch=epoch,
                train=False,
                print_every=print_every,
                seg_transformer=seg_transformer,
            )
        val_extra = ""
        if "dice_after" in val_metrics:
            val_extra += f", dice_before={val_metrics['dice_before']:.6f}, dice_after={val_metrics['dice_after']:.6f}"
        if "folding_ratio" in val_metrics:
            val_extra += f", fold={val_metrics['folding_ratio']:.6f}"
        print(
            f"Epoch {epoch}/{epochs} done: "
            f"train_total={train_metrics['total']:.6f}, "
            f"val_total={val_metrics['total']:.6f}, "
            f"val_diff={val_metrics['diff']:.6f}"
            f"{val_extra}"
        )
        if epoch % save_every == 0:
            save_checkpoint(save_dir / "last.pt", model, optimizer, epoch, best_val, model_config, config, args)
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            save_checkpoint(save_dir / "best.pt", model, optimizer, epoch, best_val, model_config, config, args)


if __name__ == "__main__":
    main()
