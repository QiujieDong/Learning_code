"""Defines the neural network, loss function and metrics"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, params):
        """Define a convolutional network.

        Args:
            params: (Params) contains num_channels
        """

        super(Net, self).__init__()
        self.num_channels = params.num_channels
        self.output_classes = params.output_classes
        self.dropout_rate = params.dropout_rate

        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.conv2 = nn.Conv2d(
            self.num_channels, self.num_channels * 2, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels * 2)
        self.conv3 = nn.Conv2d(
            self.num_channels * 2, self.num_channels * 4, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.num_channels * 4)

        self.fc1 = nn.Linear(8 * 8 * self.num_channels * 4,
                             self.num_channels * 4)
        self.fcbn1 = nn.BatchNorm1d(self.num_channels * 4)
        self.fc2 = nn.Linear(self.num_channels * 4, self.output_classes)

        self.MaxPool2d = nn.MaxPool2d(2)  # kernel_size = 2
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d()

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)

    def forward(self, s):
        """This function defines how to use the components of the network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x64 x 64

        Returns:
            out: (Variable) dimension batch_size x output_classes with the log probabilities for the labels of each image.
        """

        # Apply the convolution layers, followed by batch normalisation, maxpool and relu
        s = self.relu(self.MaxPool2d(self.bn1(self.conv1(s))))
        s = self.relu(self.MaxPool2d(self.bn2(self.conv2(s))))
        s = self.relu(self.MaxPool2d(self.bn3(self.conv3(s))))

        # flatten the output for each image
        s = s.view(s.shape[0], -1)

        # apply 2 fully connected layers with dropout. apply dropout if train=True
        s = self.dropout(self.relu(self.fcbn1(self.fc1(s))),
                         p=self.dropout_rate, inplace=True)
        s = self.fc2(s)

        return s


def loss_fn(args):
    """Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Tensor) dimension batch_size x output_classes
        labels: (Tensor) dimension batch_size

    Return:
        loss: (Tensor) cross entropy loss for all in the batch
    """

    return nn.CrossEntropyLoss().to(args.device)


def accuracy(outputs, labels):
    """Compute the accuracy, given the outputs and labels for all in the batch.

    Args:
        outputs: (Tensor) dimension batch_size x output_classes
        labels: (Tensor) dimension batch_size

    Return:
        accuracy (float) in [0, 1]
    """

    values, predicted_class = torch.max(outputs, dim=1)

    return int((predicted_class == labels).sum()) / len(labels)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # add more metrics
}
