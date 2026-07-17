from __future__ import annotations

import argparse
import dataclasses
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from hiro_llm.config import ProjectConfig, config_from_dict, load_config
from hiro_llm.data.text import (
    PretrainingDataset,
    TokenPairDataset,
    prepare_instruction_data,
    prepare_pretraining_data,
)
from hiro_llm.data.vision import VisionDataset, prepare_vision_data
from hiro_llm.evaluation import evaluate_all
from hiro_llm.inference import generate_text, generate_vision_text
from hiro_llm.model import LanguageModel, VisionLanguageModel
from hiro_llm.training.checkpoint import (
    HubPublisher,
    read_checkpoint,
    resolve_checkpoint,
)
from hiro_llm.training.logging import ExperimentLogger
from hiro_llm.training.trainer import Trainer


def _device(name: str) -> torch.device:
    if name != "auto":
        device = torch.device(name)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return device


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data(config: ProjectConfig) -> list[Path]:
    if config.task == "pretrain":
        outputs = [prepare_pretraining_data(config.data)]
    elif config.task == "instruction_tuning":
        outputs = list(prepare_instruction_data(config.data, config.model))
    else:
        outputs = list(prepare_vision_data(config.data, config.model, config.vision))
    for path in outputs:
        print(path)
    return outputs


def _language_model_from_checkpoint(
    config: ProjectConfig, reference: str | None, device: torch.device
) -> LanguageModel:
    model = LanguageModel(config.model)
    if reference:
        path = resolve_checkpoint(reference)
        payload = read_checkpoint(path, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
    return model


def _build_model(
    config: ProjectConfig, device: torch.device, load_initial_weights: bool = True
) -> torch.nn.Module:
    if config.task.startswith("vision"):
        language_checkpoint = None
        if load_initial_weights and config.task == "vision_pretrain":
            language_checkpoint = config.vision.language_model_checkpoint
        language_model = _language_model_from_checkpoint(
            config,
            language_checkpoint,
            device,
        )
        model = VisionLanguageModel(language_model, config.vision)
        model.freeze_for_stage(config.task == "vision_instruction_tuning")
    else:
        model = LanguageModel(config.model)
    if load_initial_weights and config.training.init_from:
        payload = read_checkpoint(
            resolve_checkpoint(config.training.init_from), map_location=device
        )
        model.load_state_dict(payload["model_state_dict"])
    return model.to(device)


def _build_batches(config: ProjectConfig):
    processed = Path(config.data.output_dir)
    if config.task == "pretrain":
        token_path = processed / (
            "train_tokens.npy" if (processed / "train_tokens.npy").exists() else "tokens.npy"
        )
        dataset = PretrainingDataset(token_path, config.model.max_sequence_length)
    elif config.task == "instruction_tuning":
        dataset = TokenPairDataset(processed / "input_ids.npy", processed / "labels.npy")
    else:
        dataset = VisionDataset(
            processed / "image_paths.json",
            processed / "vision_input_ids.npy",
            processed / "vision_labels.npy",
            config.vision.encoder_name,
        )
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.runtime.num_workers,
        pin_memory=config.runtime.device in {"auto", "cuda"},
    )


def _build_validation_batches(config: ProjectConfig):
    if config.task != "pretrain":
        return None
    processed = Path(config.data.output_dir)
    path = processed / "validation_tokens.npy"
    if not path.exists():
        return None
    dataset = PretrainingDataset(path, config.model.max_sequence_length)
    return DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False)


def train(config: ProjectConfig) -> Path:
    _set_seed(config.training.seed)
    device = _device(config.runtime.device)
    model = _build_model(config, device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.training.max_learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )
    if config.training.compile_model:
        model = torch.compile(model)
    stage_name = (
        "shakespeare-pretrain-10000"
        if "shakespeare" in config.data.source
        else "tinystories-adaptation"
        if "tinystories" in config.data.source.lower()
        else "pretrain"
    )
    if config.task == "instruction_tuning":
        stage_name = (
            "smoltalk-sft"
            if "smoltalk" in config.data.source.lower()
            else "instruction-tuning"
        )
    run_name = f"{config.variant}-{stage_name}" if config.variant != "baseline" else stage_name
    logger = ExperimentLogger(
        config.wandb,
        default_name=run_name,
        run_config=config.to_dict(),
        tags=[config.variant, config.task, config.data.source],
    )
    trainer = Trainer(
        model,
        optimizer,
        _build_batches(config),
        config,
        device,
        HubPublisher(config.hub),
        validation_batches=_build_validation_batches(config),
        logger=logger,
    )
    if config.training.resume_from:
        trainer.resume(resolve_checkpoint(config.training.resume_from))
    return trainer.train()


def generate(config: ProjectConfig, args: argparse.Namespace) -> str:
    device = _device(config.runtime.device)
    reference = args.checkpoint or config.training.resume_from or config.training.init_from
    if not reference:
        raise ValueError("generation requires --checkpoint or a configured checkpoint")
    model = _build_model(config, device, load_initial_weights=False)
    payload = read_checkpoint(resolve_checkpoint(reference), map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    if isinstance(model, VisionLanguageModel):
        if not args.image:
            raise ValueError("vision generation requires --image")
        return generate_vision_text(
            model,
            config.data.tokenizer,
            args.image,
            args.prompt,
            device,
            args.max_new_tokens,
            args.temperature,
        )
    return generate_text(
        model,
        config.data.tokenizer,
        args.prompt,
        device,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hiro-llm")
    subcommands = parser.add_subparsers(dest="command", required=True)
    data_parser = subcommands.add_parser("data", help="data operations")
    data_subcommands = data_parser.add_subparsers(dest="data_command", required=True)
    prepare = data_subcommands.add_parser("prepare", help="download and preprocess data")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--set", action="append", default=[], dest="overrides")
    train_parser = subcommands.add_parser("train", help="train or resume a model")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--set", action="append", default=[], dest="overrides")
    generate_parser = subcommands.add_parser("generate", help="generate text")
    generate_parser.add_argument("--config", required=True)
    generate_parser.add_argument("--checkpoint")
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--image")
    generate_parser.add_argument("--max-new-tokens", type=int, default=64)
    generate_parser.add_argument("--temperature", type=float, default=0.8)
    generate_parser.add_argument("--top-k", type=int)
    generate_parser.add_argument("--set", action="append", default=[], dest="overrides")
    evaluate_parser = subcommands.add_parser("evaluate", help="evaluate a checkpoint")
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument(
        "--benchmark", choices=["wikitext2", "ptb", "lambada", "all"], default="all"
    )
    evaluate_parser.add_argument("--config")
    evaluate_parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "evaluate" and not args.config:
        payload = read_checkpoint(resolve_checkpoint(args.checkpoint))
        config = config_from_dict(payload["config"], args.overrides)
    else:
        config = load_config(args.config, args.overrides)
    if args.command == "data":
        prepare_data(config)
    elif args.command == "train":
        checkpoint = train(config)
        print(checkpoint)
    elif args.command == "generate":
        print(generate(config, args))
    else:
        device = _device(config.runtime.device)
        names = ["wikitext2", "ptb", "lambada"] if args.benchmark == "all" else [args.benchmark]
        results = evaluate_all(args.checkpoint, names, device, config)
        for name in names:
            logger = ExperimentLogger(
                dataclasses.replace(config.wandb, run_name=f"benchmark-{name}"),
                default_name=f"benchmark-{name}",
                run_config={"checkpoint": args.checkpoint, "benchmark": name, **config.to_dict()},
                tags=["benchmark", name, config.variant],
            )
            logger.log({key: value for key, value in results.items() if key.startswith(name)}, 0)
            logger.finish()
        print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
