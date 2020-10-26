"""Evaluates the model using PyTorch."""

import os
import argparse

# use NVIDIA apex, calculate distributed data parallel, to achieve accelerate
import apex
from apex.parallel import DistributedDataParallel as DDP

# use torch
import torch
import torch.backends.cudnn as cudnn  # automatic search convolution algorithm

# use packages
import numpy as np

# add logging
import wandb  # replace logging with wandb

# use extend packages
import model.net as net
import model.data_loader as data_loader
import utils.utils as utils

paraer = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='/data/SIGNS_data/64x64_SIGNS',
                    help="Directory containing the dataset")
paraer.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing the params.json and checkpoints of train")
paraer.add_argument('--restore_weights', default=None,
                    help="Optional (best of last), restore weights file in --model_dir before training")
paraer.add_argument('--local_rank', default=-1, type=int,
                    help="Rank of the current process")
