"""Training loop for ChessNet.

Environment-agnostic: takes plain paths and a config, doesn't know or care
whether it's running locally, on Kaggle, or anywhere else. Where it runs is a
separate, thin wrapper's job -- see the local/Kaggle runner split discussed
alongside this module.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import mlflow
import mlflow.pytorch
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from chessmodel.data.dataset import (
    DEFAULT_VAL_FRACTION,
    ChessPositionDataset,
    collate_samples,
    count_games,
    split_train_val,
)
from chessmodel.data.dataset_card import estimate_avg_plies
from chessmodel.training.model import (
    DEFAULT_NUM_RESIDUAL_BLOCKS,
    DEFAULT_TRUNK_CHANNELS,
    DEFAULT_VALUE_CHANNELS,
    DEFAULT_VALUE_HIDDEN,
    ChessNet,
)

REGISTERED_MODEL_NAME = "chessmodel"


@dataclass
class TrainingConfig:
    dataset_path: Path
    checkpoint_dir: Path
    mlflow_tracking_uri: str
    experiment_name: str = "chessmodel"
    run_name: str | None = None
    val_fraction: float = DEFAULT_VAL_FRACTION
    batch_size: int = 256
    num_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    grad_clip_norm: float = 1.0
    num_workers: int = 0
    shuffle_buffer_size: int = 10_000
    max_games: int | None = None
    device: str | None = None
    resume_from: Path | None = None
    log_every_n_steps: int = 50
    seed: int = 0
    trunk_channels: int = DEFAULT_TRUNK_CHANNELS
    num_residual_blocks: int = DEFAULT_NUM_RESIDUAL_BLOCKS
    value_channels: int = DEFAULT_VALUE_CHANNELS
    value_hidden: int = DEFAULT_VALUE_HIDDEN


def _resolve_mlflow_tracking_uri(value: str) -> str:
    """Normalize a bare local path into a local SQLite-backed tracking URI.

    Two independent problems, one fix. First: MLflow's plain filesystem
    backend (`file://...`) is in maintenance mode as of MLflow 3.x and
    refuses to initialize without an explicit opt-out env var -- so a bare
    local path is treated as shorthand for "put a local SQLite database
    here," which is also MLflow's own currently-recommended local setup, not
    a workaround. Second: on Windows, a bare path like 'C:\\Users\\...' would
    otherwise get misparsed by a generic URI parser -- everything before the
    first colon looks like a scheme, so 'C' reads as the scheme instead of a
    drive letter. Real schemes (sqlite, http, https, ...) are never a single
    character, so a one-character or empty scheme means "this is actually a
    local path," not a URI already, and gets converted; anything else passes
    through unchanged.
    """
    scheme = urlparse(value).scheme
    if len(scheme) <= 1:
        db_path = Path(value).absolute() / "mlflow.db"
        return f"sqlite:///{db_path.as_posix()}"
    return value


def _resolve_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_datasets(
    dataset_path: Path,
    train_end: int,
    total_games: int,
    shuffle_buffer_size: int,
    seed: int,
) -> tuple[ChessPositionDataset, ChessPositionDataset]:
    """Train covers [0, train_end); val covers [train_end, total_games).

    Both bounds on val matter. An unbounded val_dataset (no end_game) reads
    all the way to the physical end of the PGN file, not just to the logical
    end of a max_games-capped split -- a real bug this project hit once, when
    validation silently ran against nearly the entire real dataset instead of
    a small intended slice. total_games must be the already-capped count
    (see _effective_total_games), not a raw file-wide game count.
    """
    train_dataset = ChessPositionDataset(
        dataset_path,
        end_game=train_end,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
    )
    val_dataset = ChessPositionDataset(
        dataset_path,
        start_game=train_end,
        end_game=total_games,
        shuffle_buffer_size=0,  # validation order doesn't matter and should stay stable
    )
    return train_dataset, val_dataset


def _effective_total_games(dataset_path: Path, max_games: int | None) -> int:
    """Real game count, optionally capped -- lets a smoke test use the head of
    a huge real dataset without waiting on a full-file scan's worth of training."""
    total = count_games(dataset_path)
    if max_games is not None:
        return min(total, max_games)
    return total


def _build_model(config: TrainingConfig) -> ChessNet:
    return ChessNet(
        trunk_channels=config.trunk_channels,
        num_residual_blocks=config.num_residual_blocks,
        value_channels=config.value_channels,
        value_hidden=config.value_hidden,
    )


def _estimate_total_steps(train_games: int, config: TrainingConfig) -> int:
    """Rough step-count for the LR schedule's cosine horizon.

    ChessPositionDataset streams positions lazily and deliberately doesn't
    know its own length up front (ADR-0002). Reuses dataset_card.py's
    reservoir-sampling estimator rather than duplicating that logic -- a
    ~20-30s one-time cost against a real dataset, negligible next to an
    hours-long training run, and the cosine schedule only needs to be
    roughly right (see _warmup_cosine_lr_lambda).
    """
    avg_plies, sampled = estimate_avg_plies(config.dataset_path, sample_size=2000, seed=config.seed)
    if sampled == 0:
        avg_plies = 1.0  # degenerate fallback for tiny/synthetic datasets in tests
    positions_per_epoch = train_games * avg_plies
    steps_per_epoch = max(1, int(positions_per_epoch / config.batch_size))
    return steps_per_epoch * config.num_epochs


def _warmup_cosine_lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    total_steps = max(total_steps, warmup_steps + 1)
    progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(
    path: Path,
    model: ChessNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    config: TrainingConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "config": config_dict,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: ChessNet,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    checkpoint: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def _run_validation(
    model: ChessNet,
    val_loader: DataLoader,
    device: torch.device,
    policy_loss_fn: torch.nn.Module,
    value_loss_fn: torch.nn.Module,
) -> tuple[float, float]:
    """Returns (val_loss, policy_top1_accuracy), both sample-weighted averages."""
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    with torch.no_grad():
        val_progress = tqdm(val_loader, desc="Validating", unit="batch")
        for boards, policy_targets, value_targets in val_progress:
            boards = boards.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)

            policy_logits, value = model(boards)
            loss = policy_loss_fn(policy_logits, policy_targets) + value_loss_fn(
                value, value_targets
            )

            batch_size = boards.shape[0]
            total_loss += loss.item() * batch_size
            total_correct += (policy_logits.argmax(dim=1) == policy_targets).float().sum().item()
            total_samples += batch_size

    model.train()
    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_correct / total_samples


def train(config: TrainingConfig) -> Path:
    """Train ChessNet per config. Returns the path to the best checkpoint."""
    torch.manual_seed(config.seed)
    device = _resolve_device(config.device)
    use_amp = device.type == "cuda"

    total_games = _effective_total_games(config.dataset_path, config.max_games)
    train_end, _total = split_train_val(total_games, config.val_fraction)
    train_dataset, val_dataset = _build_datasets(
        config.dataset_path, train_end, total_games, config.shuffle_buffer_size, config.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_samples,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_samples,
        num_workers=config.num_workers,
    )

    model = _build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    total_steps = _estimate_total_steps(train_end, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _warmup_cosine_lr_lambda(step, config.warmup_steps, total_steps),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    policy_loss_fn = torch.nn.CrossEntropyLoss()
    value_loss_fn = torch.nn.MSELoss()

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    if config.resume_from is not None:
        checkpoint = load_checkpoint(config.resume_from, model, optimizer, scheduler, device)
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        best_val_loss = checkpoint["best_val_loss"]

    checkpoint_dir = Path(config.checkpoint_dir)
    latest_path = checkpoint_dir / "latest.pt"
    best_path = checkpoint_dir / "best.pt"

    mlflow.set_tracking_uri(_resolve_mlflow_tracking_uri(config.mlflow_tracking_uri))
    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run(run_name=config.run_name):
        mlflow.log_params({k: str(v) for k, v in asdict(config).items()})
        mlflow.log_param("resolved_device", device.type)
        mlflow.log_param("estimated_total_steps", total_steps)

        model.train()
        for epoch in range(start_epoch, config.num_epochs):
            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{config.num_epochs}",
                unit="batch",
            )
            for boards, policy_targets, value_targets in progress:
                boards = boards.to(device)
                policy_targets = policy_targets.to(device)
                value_targets = value_targets.to(device)

                optimizer.zero_grad()
                with torch.amp.autocast(device.type, enabled=use_amp):
                    policy_logits, value = model(boards)
                    loss = policy_loss_fn(policy_logits, policy_targets) + value_loss_fn(
                        value, value_targets
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                current_lr = scheduler.get_last_lr()[0]
                progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")

                if global_step % config.log_every_n_steps == 0:
                    mlflow.log_metric("train_loss", loss.item(), step=global_step)
                    mlflow.log_metric("lr", float(scheduler.get_last_lr()[0]), step=global_step)
                global_step += 1

            val_loss, val_accuracy = _run_validation(
                model, val_loader, device, policy_loss_fn, value_loss_fn
            )
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_policy_top1_accuracy", val_accuracy, step=epoch)

            save_checkpoint(
                latest_path, model, optimizer, scheduler, epoch, global_step, best_val_loss, config
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    best_val_loss,
                    config,
                )

        mlflow.log_artifact(str(best_path))
        # log_artifact alone stores a plain file -- the registry needs a
        # properly flavor-logged model (MLmodel metadata) to register against,
        # which is what this call provides. Our own checkpoint format (via
        # save_checkpoint/load_checkpoint) stays the source of truth for
        # actually loading a model elsewhere in this codebase; this is purely
        # to get real Model Registry versioning (Phase 06's promotion gating
        # will reference it by name/version).
        mlflow.pytorch.log_model(
            model,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            serialization_format="pickle",
        )

    return best_path
