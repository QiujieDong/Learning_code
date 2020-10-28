"""Evaluates the model using PyTorch."""

import argparse
import logging
import os

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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='/data/SIGNS_data/64x64_SIGNS',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing the params.json and checkpoints of train")
parser.add_argument('--restore_weights', default=None,
                    help="Optional (best of last), restore weights file in --model_dir before training")
parser.add_argument('--local_rank', default=-1, type=int,
                    help="Rank of the current process")


def evaluate(model, loss_fn, dataloader, metrics, params, args):
    """Evaluate the model on 'num_steps' batches.

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # Set model to evaluate mode
    model.eval()

    # summary for current eval loop
    summ = []
    wandb_images_test = []

    # compute metrics over the dataset
    with torch.no_grad():
        for data_batch, labels_batch in dataloader:
            # move to device
            data_batch = data_batch.to(args.device, non_blocking=params.cuda)
            labels_batch = labels_batch.to(
                args.device, non_blocking=params.cuda)

            # compute model output
            output_batch = model(data_batch)
            loss = loss_fn(output_batch, labels_batch)

            # compute all metrics on this batch (Use predefine in Net.py)
            summary_batch = {metric: metrics[metric](
                output_batch, labels_batch) for metric in metrics}
            summary_batch['loss'] = loss.item()

            if params.distributed:
                for k, v in summary_batch.items():
                    v = utils.reduce_tensor(torch.tensor(
                        v, device=args.device), args).item()
                    summary_batch[k] = v
            summ.append(summary_batch)

            wandb_images_test.append(wandb.Image(
                data_batch[0], caption="Pred: {}\nTruth: {}".format(summary_batch['accuracy'], summary_batch['loss'])))

        # compute mean of all metrics in summary
        torch.cuda.synchronize()
        metrics_mean = {metric: np.mean([x[metric]
                                         for x in summ]) for metric in summ[0]}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                    for k, v in metrics_mean.items())

        logging.info("- Eval metrics : " + metrics_string)
        wandb.log({
            "Test wandb images": wandb_images_test,
            "Test Accuracy": 100 * metrics_mean['accuracy'],
            "Test Loss": metrics_mean['loss']})

        return metrics_mean


if __name__ == '__main__':
    # load the parameters
    args = parser.parse_args()
    # load the parameters from the params.json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)

    params = utils.Params(json_path)

    # set the logger using wandb, login first
    wandb.init(project="Qiujie_PyTorchTemplate_evaluate", config=params)

    wandb.log("Evaluates the model using PyTorch.")

    # Set random seed
    wandb.log("Set random seed={}".format(params.seed))
    utils.set_seed(params)
    wandb.log("- done")

    # Set device
    device = None
    if params.cude:
        device = torch.device('cuda')
        cudnn.benchmark = True  # Enable cudnn
    else:
        device = torch.device('cpu')
    if params.distributed and params.device_count > 1 and params.cuda:
        torch.cuda.set_device(args.lock_rank)
        device = torch.device('cuda', args.lock_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.world_size = torch.distributed.get_world_size()

    args.device = device

    # Create the input data pipeline
    wandb.log("Creating the datasets...")
    dataloaders, samplers = data_loader.fetch_dataloader(
        ['test'], args.dataset_dir, params)
    test_dataloader = dataloaders['test']
    wandb.log("- done")

    # Define the model
    wandb.log("Define model...")
    model = net.Net(params).to(args.device)

    if params.sync_bn:
        wandb.log("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    if params.distributed and params.device_count > 1 and params.cuda:
        wandb.log("- WARNING: Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                  args.lock_rank, device, params.device_count, bool(args.local_rank != -1))
        model = DDP(model)

    wandb.log("- done")

    # fetch loss function and metrics
    loss_fn = net.loss_fn(args)
    metrics = net.metrics

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(
        args.model_dir, args.restore_weights + '.pth.tar'), args, model)

    # Evaluate
    test_metrics = evaluate(
        model, loss_fn, test_dataloader, metrics, params, args)
    save_path = os.path.join(
        args.model_dir, "metrics_test_{}.json".format(args.restore_weights))

    # Save on GPU#0
    if args.local_rank in [-1, 0]:
        utils.save_dict_to_json(test_metrics, save_path)
