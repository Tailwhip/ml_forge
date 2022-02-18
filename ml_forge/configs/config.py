from dataclasses import dataclass
import os
import sys
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

from ..logger import logger


class IConfig(ABC):
    """!
    An interface class for all configs.
    """
    @abstractmethod
    def load(self, config: dict):
        pass

    @abstractmethod
    def verify(self, config: dict):
        pass

    @abstractmethod
    def generate_template(self, dir: str):
        pass