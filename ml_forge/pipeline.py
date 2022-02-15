
import os
import sys
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .configs.config_provider import ConfigProvider as CP


class IPipeline(ABC):  

    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def run(self):
        pass


class IClfTrainPipeline():

    @abstractmethod
    def load_optimizer(self):
        pass

    @abstractmethod
    def load_criterion(self):
        pass

    @abstractmethod
    def load_dataset(self):
        pass


class DefaultClfPipeline(IClfTrainPipeline, IPipeline):

    def __init__(self, data_loader: DataLoader) -> None:
        self.__model = None
        self.__data_loader = data_loader
        self.__dataset = None
        self.__criterion = None

    

