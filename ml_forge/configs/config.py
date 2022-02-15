from dataclasses import dataclass
import os
import sys
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import yaml

from ..logger import logger


class IConfig(ABC):
    
    @abstractmethod
    def load(self, config: dict):
        pass

    @abstractmethod
    def verify(self, config: dict):
        pass

    @abstractmethod
    def generate_template(self, dir: str):
        pass


class ClassificationConfig(IConfig):
    
    params = [
        "model_name",
        "weights_dir",
        "input_shape",
        "n_classes",
        "n_epochs",
        "batch_size"
    ]

    def __str__(self) -> str:
        return str(self.params)

    def load(self, config: dict):
        self.verify(config)
        self.model_name = config["model_name"]
        self.weights_dir = config["weights_dir"]
        self.input_shape = config["input_shape"]
        self.n_classes = config["n_classes"]
        self.n_epochs = config["n_epochs"]
        self.batch_size = config["batch_size"]

    def verify(self, config: dict):
        """!
        Checks if given config file has all necessary parameters.
        """
        logger.info("Verifying config...")
        for p in ClassificationConfig.params:
            if not p in config.keys():
                raise AttributeError(f"Parameter '{p}' not found "
                                      "in the config.")
        logger.info("Config verified successfully!")

    def generate_template(self, dir: str):
        template = {}

        for p in ClassificationConfig.params:
            template[p] = None

        with open(dir, "w+") as f:
            yaml.dump(template, f)