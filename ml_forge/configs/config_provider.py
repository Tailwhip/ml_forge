import os
import sys
from typing import List, Tuple, Union
import yaml

from .config import IConfig
from ..utils import SingletonMeta
from ..logger import logger


class ConfigProvider(metaclass=SingletonMeta):

    config: IConfig = None

    def __init__(self, config_dir: str) -> None:
        self.__load_config(config_dir)

    def __load_config(self, config_dir: str):
        """!
        Configuration loader.
        """
        with open(config_dir) as f:
            ConfigProvider(yaml.load(f, Loader=yaml.FullLoader))
        logger.info(f"Config loaded from dir: {config_dir}")
