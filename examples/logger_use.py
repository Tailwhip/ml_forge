#!/usr/bin/env python

import argparse
import sys

from ml_forge.logger import logger


def main():
    logger.info("Info message.")
    logger.warning("Warning message.")
    logger.debug("Debug message.")
    logger.error("Error message.")


if __name__ == "__main__":
    main()
