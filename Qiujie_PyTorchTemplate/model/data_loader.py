"""Loading data from the dataset.

borrowed from
    http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
    http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    define a training image loader that specifies transforms on images. 
    See documentation for more details.
"""

import os

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms as transforms


train_transformer = transforms.Compose(
    [transforms.RandomHorizontalFlip(), transforms.ToTensor()])

# loader for evaluation, no horizontal flip
eval_transformer = transforms.Compose([transforms.ToTensor()])
