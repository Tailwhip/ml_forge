import os
import sys
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from ..configs.config_provider import ConfigProvider as CP
from .pipeline import IClfTrainPipeline, IPipeline
from ..logger import logger


class ClassificationPipeline(IClfTrainPipeline, IPipeline):

    def __init__(self, data_loader: DataLoader) -> None:
        self.__model = None
        self.__data_loader = data_loader
        self.__dataset = None
        self.__loss_function = None
        self.__optimizer = None

    def load_model(self):
        self.__model = models.resnet18(pretrained=CP.config.pretrained)
        logger.info(f"Model has been loaded.")

    def load_optimizer(self):
        self.__optimizer = optim.SGD(self.__model.parameters(), 
                                     lr=CP.config.lr, 
                                     momentum=CP.config.momentum)
        logger.info(f"Optimizer has been loaded.")
    
    def load_loss_function(self):
        self.__loss_function = nn.CrossEntropyLoss()
        logger.info(f"Loss function has been loaded.")

    def load_dataset(self):
        self.__dataset = self.__data_loader
        logger.info(f"Dataset has been loaded.")

    def run(self):
        pass

