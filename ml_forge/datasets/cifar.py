import os
import sys
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

from ..logger import logger


class CifarDL(Dataset):

    def __init__(self, split_type: str="train"):
        self.__split_type = split_type
        self.__dataset = None
        self.__load_dataset()


    def __load_dataset(self):
        self.__dataset = torchvision.datasets.CIFAR10(root='./data', 
                                                      train=self.__split_type,
                                                      download=True, 
                                                      transform=\
                                                          self.__transform)
        logger.info(f"Dataset ({self.__split_type}) has been loaded.")


    def __getitem__(self, key):
        if torch.is_tensor(key):
            images, labels = self.__dataset[key]
            return images, labels
        else:
            raise ValueError(f"No index '{key}' in dataset!")


    def __len__(self):
        return len(self.__dataset)


    def __transform(self):
        return transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), 
                                                       (0.5, 0.5, 0.5))])