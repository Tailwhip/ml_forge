import os
import sys
from typing import List, Tuple, Union
import yaml

from .config import IConfig
from ..utils import SingletonMeta
from ..logger import logger


class ConfigProvider(metaclass=SingletonMeta):

    config: IConfig = None

    def __init__(self, config: IConfig) -> None:
        ConfigProvider.config = config

    @classmethod
    def load_config(cls, config_dir: str):
        """!
        Configuration loader.
        """
        logger.info(f"Loading config from dir: {config_dir}")
        with open(config_dir) as f:
            yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
        cls.config.load(yaml_cfg)
        logger.info(f"Config has been loaded.")
