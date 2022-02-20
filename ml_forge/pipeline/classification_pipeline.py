from cProfile import label
from distutils.command.config import config
import os
import sys
from abc import ABC, abstractmethod
from typing import List, Tuple, Union
import time

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..configs.config_provider import ConfigProvider as CP
from .pipeline import IClfTrainPipeline, IPipeline
from ..logger import logger
from ..utils import DEFAULT_DEVICE, update_pbar


class clfTrainPipeline(IClfTrainPipeline, IPipeline):


    def __init__(self, dataset: Dataset) -> None:
        self.__device = DEFAULT_DEVICE
        if (CP.config.device == "cuda") and (DEFAULT_DEVICE == "cuda"):
            torch.backends.cudnn.benchmark = True
            self.__device = CP.config.device
            
        logger.info(f"{self.__device} device is going to be used.")

        self.__model = None
        self.__data_loader = None
        self.__dataset = dataset
        self.__loss_function = None
        self.__optimizer = None


    def load_model(self):
        self.__model = models.resnet18(pretrained=CP.config.pretrained)
        self.__model.to(self.__device)
        logger.info(f"Model has been loaded.")


    def load_optimizer(self):
        if self.__model:
            self.__optimizer = optim.SGD(self.__model.parameters(), 
                                        lr=CP.config.lr, 
                                        momentum=CP.config.momentum)
            logger.info(f"Optimizer has been loaded.")
        else:
            logger.error(f"No model found. Load model first!")
    

    def load_loss_function(self):
        self.__loss_function = nn.CrossEntropyLoss()
        logger.info(f"Loss function has been loaded.")


    def prepare_dataloader(self):
        self.__data_loader = DataLoader(self.__dataset,
                                        batch_size=CP.config.batch_size,
                                        shuffle=True,
                                        num_workers=CP.config.n_workers)
        logger.info(f"Dataloader has been prepared.")


    def run(self):
        
        resume = 0
        if CP.config.resume_epoch:
            resume = CP.config.resume_epoch
        # Training epoch
        for epoch in range(resume, CP.config.n_epochs):
            self.__model.train()

            pbar = tqdm(enumerate(self.__data_loader),
                        total=len(self.__data_loader))
            pbar.set_description(f"Training {CP.config.model_name} step: ")
            start_time = time.time()

            for i, batch in pbar:
                images, labels = batch
                images = images.to(self.__device)
                labels = labels.to(self.__device)

                prepare_time = start_time-time.time()

                # preform an inference
                pred = self.__model(images)
                logger.debug(pred.shape)
                loss = self.__loss_function(pred, labels)
                loss.backward()

                # gradient accumulation
                if i % CP.config.grad_acc == 0:
                    nn.utils.clip_grad_norm_(self.__model.parameters(), 1)
                    self.__optimizer.step()
                    self.__optimizer.zero_grad(set_to_none=False)

                update_pbar(i, pbar, 
                            print_interval=CP.config.bar_update_interval)
       
