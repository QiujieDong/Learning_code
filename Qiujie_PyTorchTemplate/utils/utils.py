"""Utils in the project"""

import json
import logging
import random
import os
import shutil

import numpy as np
import torch


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
        params = Params(json_path)
        print(params.learning_rate)
        params.learning_rate = 0.5  # change the value of learning_rate in params
    ```

    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity.

    Example:
    ```
        loss_avg = RunningAverage()
        loss_avg.update(2)
        loss_avg.update(4)
        loss_avg() = 3
    ```

    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def reset(self):
        self.steps = 0
        self.total = 0

    def update(self, val, n=1):
        self.steps += n
        self.total += val * n

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved in a permanent file
    Here we save it to `model_dir/train.log`.

    Example:
    ```
        logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log

    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s:%(levelname)s:%(message)s'))
        logger.addHandler(file_handler)

        # logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of float in json file.

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file

    """

    with open(json_path, 'w') as f:
        # convert the values to float for json (it doesn't accept np.array, np.float)
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint_dir):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'.
        If is_best=True, also saves checkpoint + 'best.pth.tar'.

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint_dir: (string) folder where parameters are to be saves

    """

    filepath = os.path.join(checkpoint_dir, 'last.pth.tar')
    if not os.path.exists(checkpoint_dir):
        print("Checkpoint Directory does not exist! Making directory {}".format(
            checkpoint_dir))
        os.mkdir(checkpoint_dir)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint_dir, 'best.pth.tar'))


def load_checkpoint(checkpoint_path, args, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided.
        loads state_dict of optimizer assuming it is present in checkpoint.

    Args:
        checkpoint_path: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint

    """

    if not os.path.exists(checkpoint_path):
        raise ("File doesn't exist {}".format(checkpoint_path))

    checkpoint = None

    if args.local_rank != -1:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage.cuda(args.local_rank))

    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def set_seed(params):
    """Fix random seed."""

    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)

    if params.cuda:
        torch.cuda.manual_seed(params.seed)
    if params.device_count > 1:
        torch.cuda.manual_seed_all(params.seed)


def reduce_tensor(tensor, args):
    """average tensor with all GPU"""

    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= args.world_size

    return rt


class DataPrefetcher:
    """Optimize IO speed between CPU and GPU.

    Example:
    ```
        prefetcher = DataPrefetcher(train_loader)
        data_batch, label_batch = prefetcher.next()
        while data_batch is not None:
            do something
    ```

    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor(
            [0.485 * 255, 0.465 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopInteration:
            self.next_input = None
            self.next_target = None

            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()

        return input, target
