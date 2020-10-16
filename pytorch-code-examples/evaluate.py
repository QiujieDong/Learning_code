"""Evaluates the model"""

import argparse
import logging
import os

import apex
from apex.parallel.distributed import DistributedDataParallel as DDP
import numpy as np
import torch
import torch.backends.cudnn as cudnn

import utils
from utils import reduce_tensor
import model.net as net
import model.data_loader as data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/64x64_SIGNS',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")
parser.add_argument('--local_rank', default=-1, type=int)


def evaluate(model, loss_fn, dataloader, metrics, params, args):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    with torch.no_grad():
        for data_batch, labels_batch in dataloader:
            # move to device
            data_batch = data_batch.to(args.device, non_blocking=params.cuda)
            labels_batch = labels_batch.to(args.device, non_blocking=params.cuda)

            # compute model output
            output_batch = model(data_batch)
            loss = loss_fn(output_batch, labels_batch)

            # compute all metrics on this batch (Use predefine in Net.py)
            summary_batch = {metric: metrics[metric](output_batch, labels_batch) for metric in metrics}
            summary_batch['loss'] = loss.item()

            if params.distributed:
                for k, v in summary_batch.items():
                    v = reduce_tensor(torch.tensor(v, device=args.device), args).item()
                    summary_batch[k] = v
            summ.append(summary_batch)

        # compute mean of all metrics in summary
        torch.cuda.synchronize()
        metrics_mean = {metric: np.mean([x[metric]
                                         for x in summ]) for metric in summ[0]}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                    for k, v in metrics_mean.items())
        logging.info("- Eval metrics : " + metrics_string)
        return metrics_mean


if __name__ == '__main__':
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Set the random seed for reproducible experiments
    logging.info("Set random seed={}".format(params.seed))
    utils.set_seed(params)
    logging.info("- done.")

    # Set device
    device = None
    args.world_size = 1
    if params.cuda:
        device = torch.device('cuda')
        cudnn.benchmark = True  # Enable cudnn
    else:
        device = torch.device('cpu')
    if params.distributed and params.device_count > 1 and params.cuda:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.world_size = torch.distributed.get_world_size()

    args.device = device

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    # fetch dataloaders
    dataloaders, samplers = data_loader.fetch_dataloader(['test'], args.data_dir, params)
    test_dl = dataloaders['test']
    logging.info("- done.")

    # Define the model
    logging.info("Define model and optimizer...")
    model = net.Net(params).to(args.device)
    if params.sync_bn:
        logging.info("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    if params.distributed and params.device_count > 1 and params.cuda:
        logging.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
            args.local_rank,
            device,
            params.device_count,
            bool(args.local_rank != -1),
        )
        model = DDP(model)

    # fetch loss function and metrics
    loss_fn = net.loss_fn(args)
    metrics = net.metrics

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), args, model)

    # Evaluate
    test_metrics = evaluate(model, loss_fn, test_dl, metrics, params, args)
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))

    # Save on GPU#0
    if args.local_rank in [-1, 0]:
        utils.save_dict_to_json(test_metrics, save_path)
