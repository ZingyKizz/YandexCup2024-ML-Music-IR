import gc
import logging

import hydra
import numpy as np
import torch
import torchinfo
from dotenv import load_dotenv
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler

from csi.base.utils import init_model, seed_everything
from csi.submission import make_submission
from csi.training.data.dataset import filter_tracks
from csi.training.loop.loop import train_one_epoch
from csi.training.loop.utils import (
    clean_old_content,
    freeze_layers,
    load_fold_checkpoint,
    save_checkpoint,
)
from csi.training.metrics.ndcg import compute_ndcg

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="train_pipeline", version_base=None)
def train_pipeline(cfg: DictConfig):
    logger.info(f"Input config:\n{OmegaConf.to_yaml(cfg)}")

    seed_everything(cfg.environment.seed)

    logger.info("Reading clique2tracks")
    clique2tracks = call(
        cfg.read_clique2versions,
        _convert_="partial",
    )

    filtered_clique2tracks = call(
        cfg.read_filtered_clique2versions,
        _convert_="partial",
    )

    cliques_splits = call(cfg.split_cliques, clique2tracks, _convert_="partial")
    metrics = []
    clean_old_content(cfg.training.checkpoint_dir)
    for fold, (
        (train_track2clique, train_clique2tracks),
        (val_track2clique, val_clique2tracks),
    ) in enumerate(cliques_splits):
        if filtered_clique2tracks is not None:
            train_track2clique, train_clique2tracks = filter_tracks(
                filtered_clique2tracks, val_clique2tracks
            )
        train_dataset = call(
            cfg.train_data.dataset,
            tracks_ids=list(train_track2clique),
            track2clique=train_track2clique,
            clique2tracks=train_clique2tracks,
            _convert_="partial",
        )

        val_dataset = call(
            cfg.val_data.dataset,
            tracks_ids=list(val_track2clique),
            track2clique=val_track2clique,
            clique2tracks=val_clique2tracks,
            _convert_="partial",
        )

        logger.info(
            f"Get {fold} fold of training data,\n"
            f"train tracks={len(train_dataset)}, train clicks={len(train_clique2tracks)}\n"
            f"val tracks={len(val_dataset)}, val clicks={len(val_clique2tracks)}\n"
        )

        train_loader = instantiate(cfg.train_data.dataloader, train_dataset, _convert_="partial")
        val_loader = instantiate(cfg.val_data.dataloader, val_dataset, _convert_="partial")

        model = init_model(cfg).to(cfg.environment.device)
        if fold == 0:
            logger.info(f"Model arhitecture:\n{model}")
            logger.info(f"Model summary:\n{torchinfo.summary(model)}")

        if cfg.path_to_fold_checkpoints is not None:
            model = load_fold_checkpoint(model, cfg.path_to_fold_checkpoints, fold)

        if cfg.freeze_backbone_num_layers is not None:
            model = freeze_layers(model, cfg.freeze_backbone_num_layers, cfg.freeze_ln)

        optimizer = instantiate(
            cfg.training.optimizer,
            call(cfg.training.grouped_model_parameters, model, _convert_="partial"),
            _convert_="partial",
        )
        criterion_container = instantiate(
            cfg.training.criterion_container,
            _convert_="partial",
        )
        scheduler = instantiate(cfg.training.scheduler, optimizer, _convert_="partial")
        scaler = GradScaler() if cfg.training.mixed_precision else None
        ema = instantiate(cfg.training.ema, model.parameters(), _convert_="partial")
        patience = cfg.training.early_stopping_rounds

        best_ndcg_value = None
        num_iterations = 0
        for epoch in range(cfg.training.n_epochs):
            logger.info(f"{model.__class__.__name__} model training, epoch={epoch}")

            num_iterations = train_one_epoch(
                cfg.training,
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion_container=criterion_container,
                scheduler=scheduler,
                scaler=scaler,
                ema=ema,
                device=cfg.environment.device,
                seed=cfg.environment.seed + epoch,
                num_prev_iterations=num_iterations,
            )

            logger.info(f"Validating model, epoch={epoch}")
            if ema is None:
                ndcg_value = compute_ndcg(cfg, model, val_loader)
            else:
                with ema.average_parameters():
                    ndcg_value = compute_ndcg(cfg, model, val_loader)

            logger.info(f"Epoch={epoch}, val NDCG: {ndcg_value}")

            if best_ndcg_value is None or ndcg_value > best_ndcg_value:
                logger.info(f"NDCG has improved from {best_ndcg_value} to {ndcg_value}")
                best_ndcg_value = ndcg_value

                if ema is None:
                    clean_old_content(
                        cfg.training.checkpoint_dir, prefix=f"checkpoint__fold={fold}"
                    )
                    checkpoint_save_path = save_checkpoint(
                        cfg.training.checkpoint_dir, model, optimizer, fold, epoch, ndcg_value
                    )
                else:
                    with ema.average_parameters():
                        clean_old_content(
                            cfg.training.checkpoint_dir, prefix=f"checkpoint__fold={fold}"
                        )
                        checkpoint_save_path = save_checkpoint(
                            cfg.training.checkpoint_dir, model, optimizer, fold, epoch, ndcg_value
                        )
                logger.info(f"Model checkpoint saved to '{checkpoint_save_path}'")

                patience = cfg.training.early_stopping_rounds
            else:
                logger.info(
                    f"NDCG has not improved from {best_ndcg_value}, " f"current value={ndcg_value}"
                )
                patience -= 1
                if not patience:
                    logger.info("Early stopping condition is met. Stopped training")
                    break

            torch.cuda.empty_cache()
            gc.collect()

        metrics.append(best_ndcg_value)

    test_dataset = call(
        cfg.test_data.dataset,
        tracks_ids=np.load(cfg.test_data.test_ids_path),
        track2clique=None,
        clique2tracks=None,
        _convert_="partial",
    )
    logger.info(f"Get test data, test tracks={len(test_dataset)}")
    test_loader = instantiate(cfg.test_data.dataloader, test_dataset, _convert_="partial")

    make_submission(cfg, model, test_loader)

    logger.info("Pushing to mlflow")
    call(cfg.mlflow.push, cfg, metrics)


if __name__ == "__main__":
    load_dotenv()
    train_pipeline()
