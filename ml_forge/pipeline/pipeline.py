
import os
import sys
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

from ..configs.config_provider import ConfigProvider as CP


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
    def load_loss_function(self):
        pass

    @abstractmethod
    def load_dataset(self):
        pass




    

