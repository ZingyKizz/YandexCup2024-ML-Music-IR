import datetime
import glob
import json
import logging
from pathlib import Path

import git
import mlflow
import numpy as np
from mlflow import MlflowClient

logger = logging.getLogger(__name__)


def push_to_mlflow(
    cfg,
    metrics: list[float],
    push_config: bool = True,
    push_metrics: bool = True,
    push_model: bool = True,
    push_log: bool = True,
    push_git_info: bool = True,
    push_submission: bool = True,
) -> None:
    client = MlflowClient()
    experiment_meta = client.get_experiment_by_name(cfg.mlflow.experiment_name)
    if experiment_meta is None:
        experiment_id = client.create_experiment(cfg.mlflow.experiment_name)
    else:
        experiment_id = experiment_meta.experiment_id

    with mlflow.start_run(
        run_name=cfg.mlflow.run_name, experiment_id=experiment_id, description=cfg.description
    ):
        if push_config:
            try:
                mlflow.log_artifacts(
                    str(Path(cfg.hydra_single_run_outputs_dir) / ".hydra"), "hydra"
                )
                logger.info(
                    f"Pushed hydra config to experiment '{cfg.mlflow.experiment_name}' with id={experiment_id}"
                )
            except Exception as e:
                logger.error(f"Pushing hydra config failed with exception: {e}")

        if push_metrics:
            try:
                mlflow.log_metrics(
                    {
                        "AvgNDCG_at_100": float(np.mean(metrics)),
                        "StdNDCG_at_100": float(np.std(metrics)),
                    }
                )
                logger.info(
                    f"Pushed metrics to experiment '{cfg.mlflow.experiment_name}' with id={experiment_id}"
                )
            except Exception as e:
                logger.error(f"Pushing metrics failed with exception: {e}")

        if push_model:
            try:
                mlflow.log_artifacts(cfg.training.checkpoint_dir, "model")
                logger.info(
                    f"Pushed model checkpoint to experiment '{cfg.mlflow.experiment_name}' with id={experiment_id}"
                )
            except Exception as e:
                logger.error(f"Pushing model checkpoint failed with exception: {e}")

        if push_log:
            try:
                mlflow.log_artifact(get_log(cfg), "log")
                logger.info(
                    f"Pushed log to experiment '{cfg.mlflow.experiment_name}' with id={experiment_id}"
                )
            except Exception as e:
                logger.error(f"Pushing log failed with exception: {e}")

        if push_submission:
            try:
                mlflow.log_artifact(get_submission(cfg), "submission.csv")
                logger.info(
                    f"Pushed submission to experiment '{cfg.mlflow.experiment_name}' with id={experiment_id}"
                )
            except Exception as e:
                logger.error(f"Pushing log failed with exception: {e}")

        if push_git_info:
            try:
                mlflow.log_dict(get_git_info(), "git_info.json")
                logger.info(
                    f"Pushed git info to experiment '{cfg.mlflow.experiment_name}' with id={experiment_id}"
                )
            except Exception as e:
                logger.error(f"Pushing git info failed with exception: {e}")


def get_git_info():
    repo = git.Repo(search_parent_directories=True)
    ref = repo.head.reference
    res = {
        "branch": ref.name,
        "commit_hash": ref.commit.hexsha,
        "commit_author_name": ref.commit.author.name,
        "commit_author_email": ref.commit.author.email,
        "commit_date": str(datetime.datetime.fromtimestamp(ref.commit.committed_date)),
        "commit_message": ref.commit.message,
    }
    return res


def get_metrics(cfg):
    metrics_file = glob.glob(str(Path(cfg.training.metrics_dir) / "*"))[0]
    with open(metrics_file, mode="r") as f:
        metrics = json.load(f)
    return metrics


def get_log(cfg):
    return glob.glob(str(Path(cfg.hydra_single_run_outputs_dir) / "*.log"))[0]


def get_submission(cfg):
    return glob.glob(str(Path(cfg.submission_dir) / "submission.csv"))[0]
