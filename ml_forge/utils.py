
import os
import sys
from typing import List, Tuple, Union
import random

import torch
import numpy as np


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_random_seed(seed: int=999):
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)