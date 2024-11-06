from typing import Optional

import torch
from omegaconf import DictConfig
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage

from csi.base.utils import batch_to_device, seed_everything
from csi.training.criterion.container import CriterionContainer
from csi.training.loop.utils import RunningLoss


def train_one_epoch(
    cfg: DictConfig,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion_container: CriterionContainer,
    scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
    ema: Optional[ExponentialMovingAverage] = None,
    device: torch.device | str = "cpu",
    seed: int = 0,
    num_prev_iterations: int = 0,
) -> int:
    seed_everything(seed)

    model.train()
    model.to(device)

    running_losses = {
        criterion.__class__.__name__: RunningLoss(
            f"criterion={criterion.__class__.__name__}", total_iterations=len(train_loader)
        )
        for criterion in criterion_container
    }
    running_losses["_weighted_"] = RunningLoss(
        f"weighted loss", total_iterations=len(train_loader)
    )

    num_iterations = num_prev_iterations
    for iteration, batch in enumerate(train_loader):
        num_iterations += 1
        x = batch_to_device(batch, device)

        if cfg.mixed_precision:
            with autocast():
                outs = model(x)
        else:
            outs = model(x)

        losses = criterion_container(model, outs, batch, num_iterations)

        loss = losses["total_loss"]
        running_losses["_weighted_"].update(loss)
        for criterion_name, criterion_loss in losses["criterion_losses"].items():
            running_losses[criterion_name].update(criterion_loss)

        loss /= cfg.gradient_accumulation_steps

        if cfg.mixed_precision:
            scaler.scale(loss).backward()
            if iteration % cfg.gradient_accumulation_steps == 0:
                if cfg.gradient_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if ema is not None:
                    ema.update()

        else:
            loss.backward()

            if iteration % cfg.gradient_accumulation_steps == 0:
                if cfg.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
                optimizer.step()
                optimizer.zero_grad()

                if ema is not None:
                    ema.update()

        for loss_optimizer in losses["loss_optimizers"]:
            loss_optimizer.step()
            loss_optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

    return num_iterations
