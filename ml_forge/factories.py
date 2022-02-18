
import os
import sys
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

from torch.utils.data import DataLoader

from .configs.config_provider import ConfigProvider as CP
from .configs.classification_config import ClassificationConfig
from .pipeline.classification_pipeline import ClassificationPipeline
from .datasets.cifar import CifarDL
from .logger import logger


class IPipelineFactory(ABC):
    
    @abstractmethod
    def load_config(self):
        pass
    
    @abstractmethod
    def create_resnest18_train_pipeline(self):
        pass


class ClassificationFactory(IPipelineFactory):

    def load_config(self, config_dir: str):
        config = ClassificationConfig()
        CP(config)
        CP.load_config(config_dir)
        logger.debug(f"Loaded config: {CP.config}")
    
    def create_resnest18_train_pipeline(self):
        dataloader = CifarDL()
        pipeline = ClassificationPipeline(dataloader)
        return pipeline
    


