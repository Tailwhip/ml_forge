import os
import sys
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import yaml


class IConfig(ABC):
    
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
        "n_classes"
    ]

    def __init__(self, config: dict):
        self.verify(config)
        self.__model_name = config["model_name"]
        self.__weights_dir = config["weights_dir"]
        self.__input_shape = config["input_shape"]
        self.__n_classes = config["n_classes"]

    def verify(self, config: dict):
        for p in ClassificationConfig.params:
            if not p in config.keys():
                raise AttributeError(f"Parameter '{p}' not found "
                                      "in the config.")

    def generate_template(self, dir: str):
        template = {}

        for p in ClassificationConfig.params:
            template[p] = None

        with open(dir, "w+") as f:
            yaml.dump(template, f)