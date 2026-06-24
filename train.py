import argparse
import csv
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

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


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
    diffusion = config_section(config, "diffusion")
    if diffusion:
        if not bool(diffusion.get("enabled", True)):
            model_config["use_diffusion_scales"] = []
        elif "train_scales" in diffusion:
            model_config["use_diffusion_scales"] = diffusion.get("train_scales")
        if "timesteps" in diffusion:
            model_config["diffusion_timesteps"] = diffusion.get("timesteps")
        if "sample_steps" in diffusion:
            model_config["diffusion_sample_steps"] = diffusion.get("sample_steps")
        if "beta_schedule" in diffusion:
            model_config["diffusion_beta_schedule"] = diffusion.get("beta_schedule")
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


LOSS_KEYS = ("total", "sim", "smooth", "jac", "diff", "ms")


def _metric_value_text(name, metrics, statuses):
    status = statuses.get(name)
    if status in {"not_used", "not_computed"}:
        return status
    return f"{metrics[name]:.6f}"


def format_metrics(metrics, statuses=None):
    statuses = statuses or {}
    text = ", ".join(
        [
            f"loss={metrics['total']:.6f}",
            f"sim={metrics['sim']:.6f}",
            f"smooth={metrics['smooth']:.6f}",
            f"jac={metrics['jac']:.6f}",
            f"diff={_metric_value_text('diff', metrics, statuses)}",
            f"ms={_metric_value_text('ms', metrics, statuses)}",
        ]
    )
    if "dice_before" in metrics:
        text += f", dice_before={metrics['dice_before']:.6f}, dice_after={metrics['dice_after']:.6f}"
    if "folding_ratio" in metrics:
        text += f", fold={metrics['folding_ratio']:.6f}"
    return text


def tensor_metrics_to_float(losses):
    return {key: float(value.detach()) for key, value in losses.items() if torch.is_tensor(value)}


def get_statuses(losses):
    statuses = losses.get("statuses", {})
    return statuses if isinstance(statuses, dict) else {}


def update_metric_state(metric_state, statuses, batch_size):
    for key in ("diff", "ms"):
        status = statuses.get(key, "computed")
        if status == "computed":
            metric_state[key]["computed"] += batch_size
        else:
            metric_state[key][status] = metric_state[key].get(status, 0) + batch_size


def epoch_status(metric_state, key):
    state = metric_state.get(key, {})
    if state.get("computed", 0) > 0:
        return "computed"
    if state.get("not_computed", 0) > 0:
        return "not_computed"
    if state.get("not_used", 0) > 0:
        return "not_used"
    return "computed"


def as_list(value, length):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value for _ in range(length)]


def append_val_anomaly(csv_path, row):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "iter",
                "loss",
                "sim",
                "smooth",
                "jac",
                "fold",
                "fixed_path",
                "moving_path",
                "fixed_id",
                "moving_id",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def prefixed_summary(prefix, metrics):
    statuses = metrics.get("statuses", {})
    parts = []
    for key in ("total", "sim", "smooth", "jac", "diff", "ms"):
        display = _metric_value_text(key, metrics, statuses) if key in {"diff", "ms"} else f"{metrics[key]:.6f}"
        parts.append(f"{prefix}_{key}={display}")
    if "folding_ratio" in metrics:
        parts.append(f"{prefix}_fold={metrics['folding_ratio']:.6f}")
    return ", ".join(parts)


def progress_write(message, progress=None):
    if progress is not None and tqdm is not None:
        progress.write(message)
    else:
        print(message)


def progress_postfix(metrics, statuses, train, optimizer, best_loss=None):
    parts = [
        f"loss={metrics['total']:.4f}",
        f"sim={metrics['sim']:.4f}",
        f"jac={metrics['jac']:.2e}",
        f"diff={_metric_value_text('diff', metrics, statuses)}",
        f"ms={_metric_value_text('ms', metrics, statuses)}",
    ]
    if "folding_ratio" in metrics:
        parts.append(f"fold={metrics['folding_ratio']:.4f}")
    if train:
        parts.append(f"lr={current_lr(optimizer):.2e}")
    if best_loss is not None and best_loss < float("inf"):
        parts.append(f"best={best_loss:.4f}")
    return ", ".join(parts)


def make_progress(iterable, total, epoch, total_epochs, phase, enabled=True, leave=True):
    if not enabled or tqdm is None:
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=f"{phase} Epoch {epoch}/{total_epochs}",
        dynamic_ncols=True,
        leave=leave,
    )


def run_epoch(
    model,
    loader,
    loss_fn,
    optimizer,
    device,
    epoch,
    total_epochs,
    train=True,
    print_every=1,
    seg_transformer=None,
    logging_config=None,
    best_loss=None,
):
    model.train(train)
    logging_config = logging_config or {}
    totals = {key: 0.0 for key in LOSS_KEYS}
    metric_state = {
        "diff": {"computed": 0, "not_used": 0, "not_computed": 0},
        "ms": {"computed": 0, "not_used": 0, "not_computed": 0},
    }
    metric_totals = {"dice_before": 0.0, "dice_after": 0.0, "folding_ratio": 0.0}
    metric_count = {"dice": 0, "folding": 0}
    count = 0
    phase = "train" if train else "val"
    start_time = time.time()
    anomaly_check = (not train) and bool(logging_config.get("anomaly_check", False))
    anomaly_loss_threshold = float(logging_config.get("anomaly_loss_threshold", 0.15))
    anomaly_sim_threshold = float(logging_config.get("anomaly_sim_threshold", 0.15))
    save_val_anomalies = bool(logging_config.get("save_val_anomalies", False))
    val_anomaly_csv = logging_config.get("val_anomaly_csv", "logs/val_anomalies.csv")
    use_progress_bar = bool(logging_config.get("progress_bar", True))
    progress_leave = bool(logging_config.get("progress_leave", True))
    progress = make_progress(loader, len(loader), epoch, total_epochs, phase, enabled=use_progress_bar, leave=progress_leave)
    for step, batch in enumerate(progress, start=1):
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
        with torch.no_grad():
            fold = folding_ratio(outputs["flow"].detach())
        extra_metrics["folding_ratio"] = float(fold.detach())
        metric_totals["folding_ratio"] += extra_metrics["folding_ratio"] * batch_size
        metric_count["folding"] += batch_size
        if not train:
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
        statuses = get_statuses(losses)
        update_metric_state(metric_state, statuses, batch_size)
        for key in LOSS_KEYS:
            if key in {"diff", "ms"} and statuses.get(key) != "computed":
                continue
            totals[key] += float(losses[key].detach()) * batch_size
        if anomaly_check and (
            float(losses["total"].detach()) > anomaly_loss_threshold
            or float(losses["sim"].detach()) > anomaly_sim_threshold
        ):
            fixed_paths = as_list(batch.get("fixed_path", "unknown"), batch_size)
            moving_paths = as_list(batch.get("moving_path", "unknown"), batch_size)
            fixed_ids = as_list(batch.get("fixed_id", "unknown"), batch_size)
            moving_ids = as_list(batch.get("moving_id", "unknown"), batch_size)
            for item_idx in range(batch_size):
                row = {
                    "epoch": epoch,
                    "iter": step,
                    "loss": float(losses["total"].detach()),
                    "sim": float(losses["sim"].detach()),
                    "smooth": float(losses["smooth"].detach()),
                    "jac": float(losses["jac"].detach()),
                    "fold": extra_metrics["folding_ratio"],
                    "fixed_path": fixed_paths[item_idx],
                    "moving_path": moving_paths[item_idx],
                    "fixed_id": fixed_ids[item_idx],
                    "moving_id": moving_ids[item_idx],
                }
                progress_write(
                    "[VAL-ANOMALY] "
                    f"epoch={epoch} iter={step}/{len(loader)} "
                    f"loss={row['loss']:.6f} sim={row['sim']:.6f} "
                    f"fixed={row['fixed_path']} moving={row['moving_path']} "
                    f"fixed_id={row['fixed_id']} moving_id={row['moving_id']}",
                    progress if use_progress_bar else None,
                )
                if save_val_anomalies:
                    append_val_anomaly(val_anomaly_csv, row)
        if use_progress_bar and tqdm is not None:
            iter_metrics = tensor_metrics_to_float(losses)
            iter_metrics.update(extra_metrics)
            progress.set_postfix_str(progress_postfix(iter_metrics, statuses, train, optimizer, best_loss=best_loss))
        if print_every > 0 and (step == 1 or step % print_every == 0 or step == len(loader)):
            iter_metrics = tensor_metrics_to_float(losses)
            iter_metrics.update(extra_metrics)
            lr_text = f", lr={current_lr(optimizer):.8f}" if train else ""
            if not use_progress_bar or tqdm is None:
                print(
                    f"[{phase}] epoch={epoch} iter={step}/{len(loader)} "
                    f"{format_metrics(iter_metrics, statuses)}{lr_text}, "
                    f"time={time.time() - iter_start:.2f}s"
                )
    epoch_metrics = {}
    epoch_statuses = {}
    for key in LOSS_KEYS:
        if key in {"diff", "ms"}:
            status = epoch_status(metric_state, key)
            epoch_statuses[key] = status
            denom = max(metric_state[key].get("computed", 0), 1)
            epoch_metrics[key] = totals[key] / denom if status == "computed" else 0.0
        else:
            epoch_metrics[key] = totals[key] / max(count, 1)
    if metric_count["dice"] > 0:
        epoch_metrics["dice_before"] = metric_totals["dice_before"] / metric_count["dice"]
        epoch_metrics["dice_after"] = metric_totals["dice_after"] / metric_count["dice"]
    if metric_count["folding"] > 0:
        epoch_metrics["folding_ratio"] = metric_totals["folding_ratio"] / metric_count["folding"]
    epoch_metrics["statuses"] = epoch_statuses
    print(f"[{phase}] epoch={epoch} summary {format_metrics(epoch_metrics, epoch_statuses)}, time={time.time() - start_time:.2f}s")
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
    diffusion_cfg = config_section(config, "diffusion")
    loss_cfg = config_section(config, "loss")
    weights = loss_fn.loss_weights()
    print(
        "enabled_modules: "
        f"diffusion.enabled={bool(diffusion_cfg.get('enabled', len(model_config.get('use_diffusion_scales', [])) > 0))}, "
        f"diffusion.train_scales={model_config.get('use_diffusion_scales', [])}, "
        f"loss.multi_scale.enabled={bool(config_section(loss_cfg, 'multi_scale').get('enabled', False))}, "
        f"use_jacobian_loss={weights['jac_weight'] != 0}, "
        f"use_smooth_loss={weights['smooth_weight'] != 0}"
    )
    print(
        "loss_weights: "
        f"sim_weight={weights['sim_weight']}, "
        f"smooth_weight={weights['smooth_weight']}, "
        f"jac_weight={weights['jac_weight']}, "
        f"diff_weight={weights['diff_weight']}, "
        f"ms_weight={weights['ms_weight']}"
    )

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
    logging_config = config_section(config, "logging")
    for epoch in range(start_epoch, epochs + 1):
        print(f"Starting epoch {epoch}/{epochs}")
        train_metrics = run_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            epoch=epoch,
            total_epochs=epochs,
            train=True,
            print_every=print_every,
            logging_config=logging_config,
            best_loss=best_val,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                loss_fn,
                optimizer,
                device,
                epoch=epoch,
                total_epochs=epochs,
                train=False,
                print_every=print_every,
                seg_transformer=seg_transformer,
                logging_config=logging_config,
                best_loss=best_val,
            )
        val_extra = ""
        if "dice_after" in val_metrics:
            val_extra += f", dice_before={val_metrics['dice_before']:.6f}, dice_after={val_metrics['dice_after']:.6f}"
        print(
            f"Epoch {epoch}/{epochs} done: "
            f"{prefixed_summary('train', train_metrics)}, "
            f"{prefixed_summary('val', val_metrics)}"
            f"{val_extra}"
        )
        if epoch % save_every == 0:
            save_checkpoint(save_dir / "last.pt", model, optimizer, epoch, best_val, model_config, config, args)
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            save_checkpoint(save_dir / "best.pt", model, optimizer, epoch, best_val, model_config, config, args)


if __name__ == "__main__":
    main()
