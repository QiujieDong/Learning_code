"""Evaluates the model using PyTorch."""

import logging

# use torch
import torch

# use packages
import numpy as np

# add logging
import wandb  # replace logging with wandb

# use extend packages
import utils.utils as utils


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
