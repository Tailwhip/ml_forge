
import os
import sys
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import torchvision
import torchvision.transforms as transforms

from .configs.config_provider import ConfigProvider as CP
from .configs.classification_config import ClassificationConfig
from .pipeline.classification_pipeline import clfTrainPipeline
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
        tr = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), 
                                                        (0.5, 0.5, 0.5))])
        dataset = torchvision.datasets.CIFAR10(root='./data', 
                                               train="true",
                                               download=True, 
                                               transform=tr)
        pipeline = clfTrainPipeline(dataset)
        pipeline.load_model()
        pipeline.prepare_dataloader()
        pipeline.load_loss_function()
        pipeline.load_optimizer()
        return pipeline
    


