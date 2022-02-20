#!/usr/bin/env python

import os
import sys
import argparse

from ..logger import logger
from ..utils import DEFAULT_DEVICE, set_random_seed
from ..factories import ClassificationFactory


def main():
    logger.info("Pipeline started.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", "-c", required=False,
                        default=os.path.join(os.getcwd(), 
                                             "./config/classification.yaml"),
                        help="Path to a proper .yaml config file dedicated for"
                             "pipeline purpose.")
    args = parser.parse_args()

    set_random_seed() # Reproducibility assurance

    factory = ClassificationFactory()
    factory.load_config(args.config_dir)
    train_clf_pipeline = factory.create_resnest18_train_pipeline()
    train_clf_pipeline.run()
    logger.info("Pipeline closed.")


if __name__ == "__main__":
    main()

