"""Defines the neural network, losss function and metrics"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, params):
        """
        We define an convolutional network that predicts the sign from an image.

        Args:
            params: (Params) contains num_channels
        """
        super(Net, self).__init__() #调用父类
        self.num_channels = params.num_channels
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels) #批标准化，重新调整数据分布
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels * 2, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels * 2)
        self.conv3 = nn.Conv2d(self.num_channels * 2, self.num_channels * 4, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.num_channels * 4)
        #torch.nn.Linear(in_features, out_features, bias=True)
        self.fc1 = nn.Linear(8 * 8 * self.num_channels * 4, self.num_channels * 4) #全连接层
        self.fcbn1 = nn.BatchNorm1d(self.num_channels * 4) 
        self.fc2 = nn.Linear(self.num_channels * 4, 6) #这个out_features是class的数目

        self.dropout_rate = params.dropout_rate

        self._init_weight()

    def _init_weight(self): #初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d): #判断一个对象是否是一个已知类型
                torch.nn.init.xavier_normal_(m.weight.data) #服从正态分布，何恺明提出的初始化方法
                if m.bias is not None:
                    m.bias.data.zero_() #bias初始化为0
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.weight.data.fill_(1) #权重初始化为1
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear): #torch.nn.init.normal_(tensor, mean=0, std=1)
                torch.nn.init.normal_(m.weight.data, 0, 0.01)  # m.weight.data.normal_(0, 0.01) m.bias.data.zero_()

    def forward(self, s): #前向传播
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 64 x 64 .

        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.

        """
        #                                                  -> batch_size x 3 x 64 x 64
        # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3
        s = torch.relu(torch.max_pool2d(self.bn1(self.conv1(s)), 2)) #kernel=2
        s = torch.relu(torch.max_pool2d(self.bn2(self.conv2(s)), 2))
        s = torch.relu(torch.max_pool2d(self.bn3(self.conv3(s)), 2))

        # flatten the output for each image
        s = s.view(s.shape[0], -1)

        # apply 2 fully connected layers with dropout
        s = torch.dropout(torch.relu(self.fcbn1(self.fc1(s))), p=self.dropout_rate, train=True)
        s = self.fc2(s)

        return s


def loss_fn(args):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Tensor) dimension batch_size x 6 - output of the model
        labels: (Tensor) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns:
        loss (Tensor) cross entropy loss for all images in the batch
    """
    return nn.CrossEntropyLoss().to(args.device)


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs :
        labels :

    Returns: (float) accuracy in [0,1]
    """
    _, predicted = torch.max(outputs, dim=1)
    return int((predicted == labels).sum()) / len(labels)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
