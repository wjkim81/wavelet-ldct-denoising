"""
Low-Dose CT With a Residual Encoder-Decoder Convolutional Neural Network
"""
import os
import datetime
import torch
import torch.nn as nn

# from models.convs import common
from .base_model import BaseModel

class REDCNN(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # Network parameters
        parser.add_argument('--n_feats', type=int, default=96,
            help='number of feature maps')
        parser.add_argument('--kernel_size', type=int, default=5,
            help='kernel size of convolution layer')
        parser.add_argument('--bn', action='store_true',
            help='do batch normalization')

        return parser

    @staticmethod
    def set_savedir(opt):
        dt = datetime.datetime.now()
        date = dt.strftime("%Y%m%d-%H%M")
        dataset_name = ''
        for d in opt.datasets:
            dataset_name = dataset_name + d

        model_opt = dataset_name  + "-" + date + "-" + opt.model

        if opt.prefix != '': model_opt = opt.prefix + "-" + model_opt
        if opt.suffix != '': model_opt = model_opt + "-" + opt.suffix
        
        save_dir = os.path.join(opt.checkpoints_dir, model_opt)
        return save_dir

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.model_names = ['net']
        self.loss_name = ['loss']
        self.var_name = ['x', 'out', 'target']

        # Create model
        self.net = create_model(opt).to(self.device)
        
        # Define losses and optimizers
        if self.is_train:
            self.mse_loss_criterion = nn.MSELoss()
            self.optimizer_names = ['optimizer']
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
            self.optimizers.append(self.optimizer)

    def set_input(self, input):
        self.x = input['x'].to(self.device)
        if input['target'] is not None:
            self.target = input['target'].to(self.device)

    def forward(self):
        self.out = self.net(self.x)

    def backward(self):
        self.loss = self.mse_loss_criterion(self.out, self.target)
        self.psnr = 10 * torch.log10(1 / self.loss)
        
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        self.backward()
        self.optimizer.step()
    
    def log_loss(self, opt, phase, batch_time, iter, n_iter):
        print("{} {:.3f}s => Epoch[{}/{}]({}/{}): Loss: {:.5f}, PSNR: {:.5f}".format(
            phase, batch_time, opt.epoch, opt.n_epochs, iter, n_iter,
            self.loss.item(), self.psnr.item())
        )

def create_model(opt):
    return REDCNNModel(opt)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bn=False):
        super(ConvBlock, self).__init__()
        
        self.bn = bn
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride)
        if self.bn: self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.bn: x = self.batch_norm(x)
        out = self.relu(x)
        return out

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bn=False):
        super(DeconvBlock, self).__init__()
        
        self.bn = bn
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        if self.bn: self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, residual=None):
        x = self.deconv(x)
        if self.bn: x = self.batch_norm(x)
        if residual is not None: x = torch.add(x, residual)
        out = self.relu(x)
        return out

class REDCNNModel(nn.Module):
    def __init__(self, opt):
        super(REDCNNModel, self).__init__()

        kernel_size = opt.kernel_size
        n_channels = opt.n_channels
        n_feats = opt.n_feats
        stride = 1
        bn = opt.bn

        # Do not use shift mean!!!
        # It degrades performance of RED-CNN
        # self.shift_mean = opt.shift_mean
        # if self.shift_mean:
        #     pix_range = 1.0
        #     self.sub_mean = common.MeanShift(pix_range, n_channels=self.n_channels)
        #     self.add_mean = common.MeanShift(pix_range, n_channels=self.n_channels, sign=1)

        self.conv1 = ConvBlock(n_channels, n_feats, kernel_size=kernel_size, stride=stride, bn=bn)
        self.conv2 = ConvBlock(n_feats, n_feats, kernel_size=kernel_size, stride=stride, bn=bn)
        self.conv3 = ConvBlock(n_feats, n_feats, kernel_size=kernel_size, stride=stride, bn=bn)
        self.conv4 = ConvBlock(n_feats, n_feats, kernel_size=kernel_size, stride=stride, bn=bn)
        self.conv5 = ConvBlock(n_feats, n_feats, kernel_size=kernel_size, stride=stride, bn=bn)

        self.deconv1 = DeconvBlock(n_feats, n_feats, kernel_size=kernel_size, stride=stride, bn=bn)
        self.deconv2 = DeconvBlock(n_feats, n_feats, kernel_size=kernel_size, stride=stride, bn=bn)
        self.deconv3 = DeconvBlock(n_feats, n_feats, kernel_size=kernel_size, stride=stride, bn=bn)
        self.deconv4 = DeconvBlock(n_feats, n_feats, kernel_size=kernel_size, stride=stride, bn=bn)
        self.deconv5 = DeconvBlock(n_feats, n_channels, kernel_size=kernel_size, stride=stride, bn=bn)

    def forward(self, x):
        # if self.shift_mean:
        #     x = self.sub_mean(x)
        residual1 = x
        x = self.conv1(x)
        x = self.conv2(x)
        residual2 = x
        x = self.conv3(x)
        x = self.conv4(x)
        residual3 = x
        x = self.conv5(x)

        x = self.deconv1(x, residual3)
        x = self.deconv2(x)
        x = self.deconv3(x, residual2)
        x = self.deconv4(x)
        out = self.deconv5(x, residual1)

        # if self.shift_mean:
        #     out = self.add_mean(out)

        return out