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
import torchvision.transforms as transforms
from PIL import Image


train_transformer = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])

# loader for evaluation, no horizontal flip
eval_transformer = transforms.Compose([
    transforms.ToTensor()])


class BaseDataset(Dataset):
    """A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.

    """

    def __init__(self, dataset_dir, transform):
        """Store the filenames of the signals(images, text, and so on) to use. Specifies transforms to apply on images.

        Args:
            dataset_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image

        """

        self.filenames = os.listdir(dataset_dir)
        self.filenames = [os.path.join(dataset_dir, f)
                          for f in self.filenames if f.endswith('.jpg')]
        self.labels = [int(os.path.split(filename)[-1][0])
                       for filename in self.filenames]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        """Fetch index idx signal and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Return:
            image: (Tensor) transformed image
            label: (int) corresponding label of image

        """

        image = Image.open(self.filenames[idx])
        image = self.transform(image)

        return image, self.labels[idx]


def fetch_dataloader(types, dataset_dir, params):
    """Fetches the DataLoader object for each type in types from dataset_dir.

    Args:
        types: [list] has one or more of 'train', 'val', 'test' depending on which data is required
        dataset_dir: [string] directory containing the dataset
        params: (Params) hyperparameters

    Return:
        data: (dict) contains the DataLoader object for each type in types

    """

    dataloaders = {}
    samplers = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(dataset_dir, "{}".format(split))

            # Use the train_transformer if training data, else use eval_transformer without random flip
            # take care of 'pin_memory' and 'num_workers'
            if split == 'train':
                train_set = BaseDataset(path, train_transformer)
                sampler = None
                if params.distributed:
                    sampler = torch.utils.data.distributed.DistributedSampler(
                        train_set)
                dataloader = DataLoader(
                    train_set,
                    batch_size=params.batch_size_pre_gpu,
                    shuffle=(sampler is None),
                    num_workers=params.num_workers,
                    pin_memory=params.cuda,
                    sampler=sampler)

            else:
                val_set = BaseDataset(path, eval_transformer)
                sampler = None
                if params.distributed:
                    sampler = torch.utils.data.distributed.DistributedSampler(
                        val_set)
                dataloader = DataLoader(
                    val_set,
                    batch_size=params.batch_size_pre_gpu,
                    shuffle=False,
                    pin_memory=params.cuda,
                    num_workers=params.num_workers,
                    sampler=sampler)

            dataloaders[split] = dataloader
            samplers[split] = sampler

    return dataloaders, samplers
