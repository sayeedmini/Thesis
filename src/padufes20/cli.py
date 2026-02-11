from __future__ import annotations

import argparse
from pathlib import Path

import torch
from rich.console import Console
from torch.utils.data import DataLoader

from padufes20.config import Config
from padufes20.data.padufes20 import PADUFES20Dataset, load_metadata
from padufes20.data.splits import SplitConfig, assert_no_group_leakage, make_patientwise_splits
from padufes20.eval.calibration import expected_calibration_error
from padufes20.eval.infer import InferConfig, predict_proba
from padufes20.eval.metrics import compute_metrics
from padufes20.eval.plots import plot_confusion_matrix, plot_reliability_diagram
from padufes20.models.registry import ModelConfig, create_model
from padufes20.training.engine import TrainConfig, train_loop
from padufes20.utils.io import ensure_dir, write_json
from padufes20.utils.seed import SeedConfig, seed_everything

console = Console()


def _dataloader(ds: PADUFES20Dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def cmd_train(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config)

    run_dir = ensure_dir(args.run_dir)
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "metrics")
    ensure_dir(run_dir / "figures")

    seed_cfg = SeedConfig(seed=int(cfg.get("seed", 42)), deterministic=bool(cfg.get("deterministic", True)))
    seed_everything(seed_cfg)

    meta = load_metadata(args.data_root)
    split_cfg = SplitConfig(
        test_size=float(cfg.get("split.test_size", 0.2)),
        val_size=float(cfg.get("split.val_size", 0.1)),
        seed=int(cfg.get("seed", 42)),
    )
    split = make_patientwise_splits(meta.df, meta.label_col, meta.patient_col, split_cfg)
    assert_no_group_leakage(meta.df, split, meta.patient_col)

    img_size = int(cfg.get("data.image_size", 224))
    ds_train = PADUFES20Dataset(meta, split["train"], args.data_root, image_size=img_size, augment=True)
    ds_val = PADUFES20Dataset(meta, split["val"], args.data_root, image_size=img_size, augment=False)
    ds_test = PADUFES20Dataset(meta, split["test"], args.data_root, image_size=img_size, augment=False)

    train_cfg = TrainConfig(
        epochs=int(cfg.get("train.epochs", 20)),
        batch_size=int(cfg.get("train.batch_size", 32)),
        lr=float(cfg.get("train.lr", 3e-4)),
        weight_decay=float(cfg.get("train.weight_decay", 1e-4)),
        num_workers=int(cfg.get("train.num_workers", 2)),
        device=str(cfg.get("train.device", "cuda")),
        amp=bool(cfg.get("train.amp", True)),
        early_stop_patience=int(cfg.get("train.early_stop_patience", 5)),
    )

    train_loader = _dataloader(ds_train, train_cfg.batch_size, train_cfg.num_workers, shuffle=True)
    val_loader = _dataloader(ds_val, train_cfg.batch_size, train_cfg.num_workers, shuffle=False)
    test_loader = _dataloader(ds_test, train_cfg.batch_size, train_cfg.num_workers, shuffle=False)

    model_cfg = ModelConfig(
        name=str(cfg.get("model.name", "resnet50")),
        pretrained=bool(cfg.get("model.pretrained", True)),
        dropout=float(cfg.get("model.dropout", 0.0)),
    )
    model = create_model(model_cfg, num_classes=len(meta.classes))

    cfg.save_resolved(run_dir / "config_resolved.yaml")

    console.print(f"[bold]Classes:[/bold] {meta.classes}")
    console.print(
        f"[bold]Split sizes:[/bold] train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}"
    )
    console.print(f"[bold]Model:[/bold] {model_cfg.name} pretrained={model_cfg.pretrained} dropout={model_cfg.dropout}")

    best_ckpt, hist = train_loop(model, train_loader, val_loader, train_cfg, ckpt_dir=run_dir / "checkpoints")
    console.print(f"[green]Best checkpoint:[/green] {best_ckpt} (best_val_loss={hist['best_val_loss']:.4f})")

    # Evaluate best checkpoint on test
    cmd_eval(
        argparse.Namespace(
            config=args.config,
            data_root=args.data_root,
            ckpt=str(best_ckpt),
            out=str(run_dir / "eval"),
        )
    )


def cmd_eval(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config)
    out_dir = ensure_dir(args.out)

    meta = load_metadata(args.data_root)
    split_cfg = SplitConfig(
        test_size=float(cfg.get("split.test_size", 0.2)),
        val_size=float(cfg.get("split.val_size", 0.1)),
        seed=int(cfg.get("seed", 42)),
    )
    split = make_patientwise_splits(meta.df, meta.label_col, meta.patient_col, split_cfg)
    assert_no_group_leakage(meta.df, split, meta.patient_col)

    img_size = int(cfg.get("data.image_size", 224))
    ds_test = PADUFES20Dataset(meta, split["test"], args.data_root, image_size=img_size, augment=False)

    train_cfg = TrainConfig(
        epochs=int(cfg.get("train.epochs", 20)),
        batch_size=int(cfg.get("train.batch_size", 32)),
        lr=float(cfg.get("train.lr", 3e-4)),
        weight_decay=float(cfg.get("train.weight_decay", 1e-4)),
        num_workers=int(cfg.get("train.num_workers", 2)),
        device=str(cfg.get("train.device", "cuda")),
        amp=bool(cfg.get("train.amp", True)),
        early_stop_patience=int(cfg.get("train.early_stop_patience", 5)),
    )
    test_loader = _dataloader(ds_test, train_cfg.batch_size, train_cfg.num_workers, shuffle=False)

    model_cfg = ModelConfig(
        name=str(cfg.get("model.name", "resnet50")),
        pretrained=False,
        dropout=float(cfg.get("model.dropout", 0.0)),
    )
    model = create_model(model_cfg, num_classes=len(meta.classes))

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)

    y_true, y_prob = predict_proba(model, test_loader, InferConfig(device=train_cfg.device))

    artifacts = compute_metrics(y_true=y_true, y_prob=y_prob, class_names=meta.classes)
    calib = expected_calibration_error(y_true=y_true, y_prob=y_prob, n_bins=int(cfg.get("eval.ece_bins", 15)))

    ensure_dir(out_dir / "metrics")
    ensure_dir(out_dir / "figures")

    scalar = dict(artifacts.scalar)
    scalar.update({"ece": calib.ece, "mce": calib.mce})

    write_json(out_dir / "metrics" / "metrics.json", scalar)
    artifacts.per_class.to_csv(out_dir / "metrics" / "per_class.csv", index=False)

    plot_confusion_matrix(artifacts.confusion, meta.classes, out_dir / "figures" / "confusion_norm.png", normalize=True)
    plot_confusion_matrix(artifacts.confusion, meta.classes, out_dir / "figures" / "confusion_counts.png", normalize=False)
    plot_reliability_diagram(calib.bin_conf, calib.bin_acc, calib.bin_count, out_dir / "figures" / "reliability.png")

    console.print("[bold]Saved metrics to:[/bold]", str(out_dir / "metrics" / "metrics.json"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="padufes20")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Train + evaluate")
    t.add_argument("--config", required=True)
    t.add_argument("--data-root", required=True)
    t.add_argument("--run-dir", required=True)
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("eval", help="Evaluate a checkpoint")
    e.add_argument("--config", required=True)
    e.add_argument("--data-root", required=True)
    e.add_argument("--ckpt", required=True)
    e.add_argument("--out", required=True)
    e.set_defaults(fn=cmd_eval)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
