"""Sample configuration dataclass for training repo."""
from typing import List, Optional, Literal, Union
from dataclasses import dataclass, field
import torch
import numpy as np
from pathlib import Path
import pyrallis
from enum import Enum


@dataclass()
class LogConfig:
    """config for logging specification"""

    # mlflow tracking uri
    uri: Optional[str] = "/data/zgazak/coco/mlruns"
    # toggle asynchronous logging (not compatible with ray tune)
    enable_async: bool = True
    # number of threads to use in async logging (2 threads/core typically)
    num_threads: int = 4
    # every `train_freq` steps, log training quantities (metrics, image batch, etc.)
    train_freq: int = 100
    # every `test_freq` steps, log test quantities (metrics, single image batch, etc.)
    val_freq: int = 500
    # every `save_freq` steps save model checkpoint according to save criteria
    save_freq: int = 1000
    # save initial model state
    save_init: bool = True
    # save last model state
    save_last: bool = True
    # save latest model
    save_latest: bool = False
    # save best model state (early stopping)
    save_best: bool = True
    # log images
    images: bool = True
    # log histograms of trainable parameters
    params: bool = False
    # log histograms of gradients
    gradients: bool = False
    # log spectral plots
    spectra: bool = False
    # save plot of p(R|x) for each image x in batch
    plot_pdf: bool = False
    # number of images to include in pdf grid plot
    n_pdf_samples: int = 6


@dataclass()
class TrainConfig:
    """config for training instance"""

    # config for logging specification
    log: LogConfig = field(default_factory=LogConfig)
    # run name
    run_name: str = "run_0"
    # experiment name
    exp_name: str = "debug"
    # gpu list to expose to training instance
    gpus: List[int] = field(default_factory=lambda: [-1])
    # random seed, set to make deterministic
    seed: int = 1983
    # number of cpu workers in dataloader
    num_workers: int = 4


if __name__ == "__main__":
    """test the train config, export to yaml"""
    cfg = pyrallis.parse(config_class=TrainConfig)
    pyrallis.dump(cfg, open("train_cfg.yaml", "w"))
