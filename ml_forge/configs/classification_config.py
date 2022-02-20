from dataclasses import dataclass
import os
import sys
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import yaml

from .config import IConfig
from ..logger import logger


class ClassificationConfig(IConfig):
    
    params = [
        "model_name",
        "weights_dir",
        "pretrained",
        "device",
        
        "n_epochs",
        "resume_epoch",
        "batch_size",
        "grad_acc",

        "dataset_name",
        "input_shape",
        "n_classes",
        "n_workers",

        "lr",
        "momentum",

        "bar_update_interval",
        "mlflow_update_interval"
    ]

    def __str__(self) -> str:
        return str(self.params)


    def load(self, config: dict):
        self.verify(config)

        self.model_name = config["model_name"]
        self.weights_dir = config["weights_dir"]
        self.pretrained = config["pretrained"]
        self.device = config["device"]

        self.n_epochs = config["n_epochs"]
        self.resume_epoch = config["resume_epoch"]
        self.batch_size = config["batch_size"]
        self.grad_acc = config["grad_acc"]

        self.dataset_name = config["dataset_name"]
        self.input_shape = config["input_shape"]
        self.n_classes = config["n_classes"]
        self.n_workers = config["n_workers"]

        self.lr = config["lr"]
        self.momentum = config["momentum"]

        self.bar_update_interval = config["bar_update_interval"]
        self.mlflow_update_interval = config["mlflow_update_interval"]


    def verify(self, config: dict):
        """!
        Checks if given config file has all necessary parameters.
        """
        logger.info("Verifying config...")
        for p in ClassificationConfig.params:
            if not p in config.keys():
                raise AttributeError(f"Parameter '{p}' not found "
                                      "in the config.")
        logger.info("Config has been verified.")


    def generate_template(self, dir: str):
        template = {}

        for p in ClassificationConfig.params:
            template[p] = None

        with open(dir, "w+") as f:
            yaml.dump(template, f)