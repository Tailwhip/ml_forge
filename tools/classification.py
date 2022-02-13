#!/usr/bin/env python

import os
import sys
import argparse

from ml_forge.logger import logger
from ml_forge.utils import set_random_seed


def main(args):
    config_dir = args.config_dir
    set_random_seed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_dir", 
                        default=os.path.join(os.getcwd(), 
                                             "../config/classification.yaml"),
                        help="Path to a proper .yaml config file dedicated for"
                             "pipeline purpose.")
    logger.info("Pipeline start.")
    args = parser.parse_args()
    main(args)
    logger.info("Pipeline close.")
