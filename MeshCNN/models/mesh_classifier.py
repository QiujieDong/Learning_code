import torch
from . import networks
from os.path import join
from util.util import seg_accuracy, print_network


class ClassifierModel:
    """ Class for training Model weights

    :args opt: structure containing configuration params
    e.g.,
    --dataset_mode -> classification / segmentation
    --arch -> network type

    """

    def __init__(self, opt):
        """初始化"""
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        self.optimizer = None  # 优化器
        self.edge_features = None  # edge特征
        self.labels = None  # Label数据
        self.mesh = None  # mesh数据
        self.soft_label = None  # soft-label数据
        self.loss = None  # loss函数

        self.nclasses = opt.nclasses  # 从label数据中得到the number of classes

        # load/define the networks
        self.net = networks.define_classifier(opt.input_nc, opt.ncf, opt.ninput_edges, opt.nclasses, opt,
                                              self.gpu_ids, opt.arch, opt.init_type, opt.init_gain)
        self.net.train(self.is_train)
        self.criterion = networks.define_loss(opt).to(self.device)  # loss的定义

        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr,
                                              betas=(opt.beta1, 0.999))  # 通过net.parameters()获取网络参数
            self.scheduler = networks.get_scheduler(self.optimizer, opt)  # lr的变化策略
            print_network(self.net)  # 打印出所有网络参数

        if not self.is_train or opt.continue_train:  # continue training: load the latest model
            self.load_network(opt.which_epoch)  # which epoch to load? set to latest to use latest cached model

    def set_input(self, data):
        """输入初始化赋值"""
        input_edge_features = torch.from_numpy(data['edge_features']).float()  # 将numpy数据转成tensor数据，并且二者共享内存。
        labels = torch.from_numpy(data['label']).long()
        self.edge_features = input_edge_features.to(self.device).requires_grad_(self.is_train)  # requires_grad_()是否自动求导
        self.labels = labels.to(self.device)
        self.mesh = data['mesh']
        if self.opt.dataset_mode == 'segmentation' and not self.is_train:
            self.soft_label = torch.from_numpy(data['soft_label'])  # 这意思是segmentation进行test时计算的是soft_label

    def forward(self):
        out = self.net(self.edge_features, self.mesh)
        return out

    def backward(self, out):
        self.loss = self.criterion(out, self.labels)
        self.loss.backward()  # 反向传播求解梯度

    def optimize_parameters(self):
        self.optimizer.zero_grad()  # 反向传播前梯度清0，如若不显示的进 optimizer.zero_grad()这一步操作，backward()的时候就会累加梯度
        out = self.forward()
        self.backward(out)  # 反向传播求解梯度
        self.optimizer.step()  # 更新权重参数

    def load_network(self, which_epoch):
        """load model from disk"""
        save_filename = '%s_net,pth' % which_epoch
        load_path = join(self.save_dir, save_filename)
        net = self.net
        if isinstance(net, torch.nn.DataParallel):  # isinstance若net参数是DataParallel(在module之间实现数据并行)的实例则返回true
            net = net.module  # 多GPU时，module将数据分发给每个GPU。
        print('loading the model from %s' % load_path)
        # PyTorch newer than 0.4 (e.g., built from GitHub source, you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.device))  # state_dict状态字典，包含整个module的所有状态的字典
        # hasattr()函数用于判断对象是否包含对应的属性。metadata(元数据)为描述数据的数据，主要描述数据属性的信息，用来支持资源查找等功能。
        # 模型部署时，要进行模型和网络精简，删除模型中无用层权重参数，注释掉无用层。否则load_state_dict 且 strict=True 时报错或者警告。
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        net.load_state_dict(state_dict)  # 从buffers和state_dict中复制parameters到module和descendants中

    def save_network(self, which_epoch):
        """save model to disk"""
        save_filename = '%s_net.pth' % (which_epoch)
        save_path = join(self.save_dir, save_filename)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict(), save_path)  # 使用多GPU训练，这里是net.module.
            self.net.cuda(self.gpu_ids[0])  # 将多GPU上的model参数和buffers移动到gpu[0]上
        else:
            torch.save(self.net.cpu().state_dict(), save_path)

    def update_learning_rate(self):
        """update learning rate (called once every )"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']  # param_groups - a dict containing all parameter groups
        print('learning rate = %.7f' % lr)

    def test(self):
        """test model
        return: number correct and total number
        """
        with torch.no_grad():  # no_grad()禁用梯度计算
            out = self.forward()
            pred_class = out.data.max(1)[1]  # (1)表示在行上求最大值，[1]表示只返回索引。也就是返回最大元素在这一行的列索引
            label_class = self.labels
            self.export_segmentation(pred_class.cpu())  # 这一句是为后面导出分割结果，与计算准确率无关
            correct = self.get_accuracy(pred_class, label_class)
        return correct, len(label_class)

    def get_accuracy(self, pred, labels):
        """computes accuracy for classification / segmentation"""
        if self.opt.dataset_mode == 'classification':
            correct = pred.eq(labels).sum  # eq()判断对象是否相等
        elif self.opt.dataset_mode == 'segmentation':
            correct = seg_accuracy(pred, self.soft_label, self.mesh)  # seg_accuracy是分割准确率的计算方法
        return correct

    def export_segmentation(self, pred_seg):  # 进行mesh输出结果的相关操作
        if self.opt.dataset_mode == 'segmentation':
            for meshi, mesh in enumerate(self.mesh):  # enumerate()同时列出索引和数据，这里meshi为索引，mesh为数据。一行mesh数据一处理。
                mesh.export_segments(pred_seg[meshi, :])
