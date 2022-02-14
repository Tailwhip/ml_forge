
import os
import sys
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

from .configs.config_provider import ConfigProvider as CP
from .logger import logger


class IPipelineFactory(ABC):
    
    @abstractmethod
    def load_config(self):
        pass
    
    @abstractmethod
    def create_pipeline(self):
        pass


class ClassificationFactory(IPipelineFactory):

    def __init__(self, config_dir: str):
        self.__config_dir = config_dir
        
    def load_config(self):
        CP(self.__config_dir)
    
    def create_pipeline(self):
        logger.debug(CP.config)
        return None
    

