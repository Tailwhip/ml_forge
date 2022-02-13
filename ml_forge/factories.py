
import os
import sys
from abc import ABC, abstractmethod
from typing import List, Tuple, Union


class IPipelineFactory(ABC):
    
    @abstractmethod
    def load_config():
        pass
    
    @abstractmethod
    def create_pipeline():
        pass
    

class ClassificationFactory(IPipelineFactory):

    def __init__(self, config_dir: str):
        self.__config_dir = config_dir
        
    def load_config():
        pass        
    

