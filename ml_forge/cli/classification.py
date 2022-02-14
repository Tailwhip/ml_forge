#!/usr/bin/env python

import os
import sys
import argparse

from ..logger import logger
from ..utils import set_random_seed
from ..factories import ClassificationFactory


def main():
    logger.info("Pipeline start.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", "-c", required=False,
                        default=os.path.join(os.getcwd(), 
                                             "./config/classification.yaml"),
                        help="Path to a proper .yaml config file dedicated for"
                             "pipeline purpose.")
    args = parser.parse_args()

    set_random_seed()

    factory = ClassificationFactory(args.config_dir)
    factory.load_config()
    classification_pipeline = factory.create_pipeline()

    logger.info("Pipeline close.")


if __name__ == "__main__":
    main()

