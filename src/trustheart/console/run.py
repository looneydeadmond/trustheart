from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from jsonargparse import ActionConfigFile, ArgumentParser
from torch import nn

from ..acquisition.beats import BeatDataset
from ..acquisition.sources import WfdbSource
from ..acquisition.splits import fraction_split
from ..acquisition.synthetic import SyntheticEcg
from ..assembly.config import DataConfig, ExperimentConfig
from ..assembly.model import TrustHeart
from ..bench.attacks import pgd
from ..bench.detection import detection_rates
from ..bench.footprint import footprint_report
from ..bench.interpretability import iou, pointing_game, saliency_over_signal
from ..bench.robustness import evaluate_attack, evaluate_clean
from ..instrument.device import select_device
from ..instrument.logs import get_logger
from ..instrument.seeding import set_seed
from ..scheduler.checkpoint import restore_model, save_model
from ..scheduler.loop import train
from ..watchdog.detector import AnomalyWatchdog

logger = get_logger("trustheart.console")


def build_datasets(data: DataConfig) -> tuple[BeatDataset, BeatDataset]:
    if data.source == "wfdb":
        beats = WfdbSource(data.root, data.sampling_rate).beats()
    else:
        beats = SyntheticEcg(data.n_classes, data.per_class, data.seed).beats()
    train_beats, test_beats = fraction_split(beats, data.train_fraction)
    return BeatDataset(train_beats), BeatDataset(test_beats)


def _annotation_mask(length: int, batch: int) -> torch.Tensor:
    mask = torch.zeros(batch, length)
    lo = int(length * 0.4)
    hi = int(length * 0.6)
    mask[:, lo:hi] = 1.0
    return mask


def run_train(exp: ExperimentConfig) -> dict[str, Any]:
    set_seed(exp.train.seed)
    device = select_device(exp.train.device)
    train_ds, _ = build_datasets(exp.data)
    model = TrustHeart(exp.model)
    state = train(model, train_ds, exp, device)
    out_dir = Path(exp.train.out_dir)
    save_model(model, exp.train.seed, out_dir / "model.pt")
    report = footprint_report(model)
    logger.info(
        "footprint params=%d int8_kb=%.1f", int(report["parameters"]), report["int8_size_kb"]
    )
    return {"final_loss": state.history[-1] if state.history else float("nan"), "steps": state.step}


def run_evaluate(exp: ExperimentConfig, checkpoint: str) -> dict[str, Any]:
    model, _ = restore_model(checkpoint)
    _, test_ds = build_datasets(exp.data)
    x, y = test_ds.signals, test_ds.labels
    clean = evaluate_clean(model, x, y)
    pgd_metrics = evaluate_attack(model, x, y, "pgd")
    logger.info("clean_acc=%.4f pgd_acc=%.4f", clean["accuracy"], pgd_metrics["accuracy"])
    return {"clean": clean, "pgd": pgd_metrics}


def run_detect(exp: ExperimentConfig, checkpoint: str) -> dict[str, Any]:
    model, _ = restore_model(checkpoint)
    train_ds, test_ds = build_datasets(exp.data)
    model.eval()
    with torch.no_grad():
        train_emb = model.embed(train_ds.signals)
    watch = AnomalyWatchdog(
        exp.model.n_classes, beta=exp.model.ladl_beta, components=exp.model.ladl_components
    )
    watch.fit(train_emb, train_ds.labels)
    x, y = test_ds.signals, test_ds.labels
    with torch.no_grad():
        clean_emb = model.embed(x)
        clean_pred = model.logits(x).argmax(dim=-1)
    clean_scores = watch.score(clean_emb, clean_pred)
    watch.calibrate(clean_scores)
    adv_x = pgd(model, x, y)
    with torch.no_grad():
        adv_emb = model.embed(adv_x)
        adv_pred = model.logits(adv_x).argmax(dim=-1)
    adv_scores = watch.score(adv_emb, adv_pred)
    rates = detection_rates(clean_scores, adv_scores, watch.threshold)
    logger.info("adr=%.4f fpr=%.4f", rates["adr"], rates["fpr"])
    return rates


def run_explain(exp: ExperimentConfig, checkpoint: str) -> dict[str, Any]:
    model, _ = restore_model(checkpoint)
    _, test_ds = build_datasets(exp.data)
    x = test_ds.signals[:32]
    y = test_ds.labels[:32]
    saliency = saliency_over_signal(model, x, y)
    mask = _annotation_mask(x.shape[-1], x.shape[0])
    result = {"iou": iou(saliency, mask), "pointing_game": pointing_game(saliency, mask)}
    logger.info("iou=%.4f pointing_game=%.4f", result["iou"], result["pointing_game"])
    return result


class LogitsModule(nn.Module):
    def __init__(self, model: TrustHeart) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.logits(x)


def run_export(exp: ExperimentConfig, checkpoint: str, path: str) -> dict[str, Any]:
    model, _ = restore_model(checkpoint)
    model.eval()
    wrapper = LogitsModule(model)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, exp.model.in_channels, 256)
    torch.onnx.export(
        wrapper, (dummy,), path, input_names=["ecg"], output_names=["logits"], opset_version=18
    )
    logger.info("exported onnx to %s", path)
    return {"path": path}


def _subparser() -> ArgumentParser:
    base = ArgumentParser()
    base.add_argument("--config", action=ActionConfigFile)
    base.add_class_arguments(ExperimentConfig, "experiment")
    return base


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="trustheart", description="TrustHeart command console")
    subcommands = parser.add_subcommands(dest="command")
    train_parser = _subparser()
    subcommands.add_subcommand("train", train_parser)
    for name in ("evaluate", "detect", "explain", "export"):
        sub = _subparser()
        sub.add_argument("--checkpoint", type=str, default="runs/main/model.pt")
        if name == "export":
            sub.add_argument("--path", type=str, default="runs/main/trustheart.onnx")
        subcommands.add_subcommand(name, sub)
    return parser


def main(args: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    cfg = parser.parse_args(args)
    init = (
        parser.instantiate_classes(cfg)
        if not hasattr(parser, "instantiate")
        else parser.instantiate(cfg)
    )
    command = init.command
    branch = init[command]
    exp: ExperimentConfig = branch.experiment
    if command == "train":
        return run_train(exp)
    if command == "evaluate":
        return run_evaluate(exp, branch.checkpoint)
    if command == "detect":
        return run_detect(exp, branch.checkpoint)
    if command == "explain":
        return run_explain(exp, branch.checkpoint)
    if command == "export":
        return run_export(exp, branch.checkpoint, branch.path)
    raise ValueError(f"unknown command {command}")


if __name__ == "__main__":
    main()
