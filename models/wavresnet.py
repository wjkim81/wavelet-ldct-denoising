"""
Deep Convolutional Framelet Denosing for Low-Dose CT via Wavelet Residual Network
https://ieeexplore.ieee.org/document/8332971
This is an advanced version of WavResNet

Wavelet Domain Residual Network (WavResNet)for Low-Dose X-ray CT Reconstruction
https://arxiv.org/pdf/1703.01383.pdf

The code is originally implemented with MATLAB, but the implementation can be found in another application of their papaers
https://github.com/eunh/CycleGAN_CT/blob/master/models/networks.py
"""
import os
import datetime
import functools

import torch
import torch.nn as nn

from models.convs.wavelet import serialize_swt, unserialize_swt
from models.convs.wavelet import SWTForward, SWTInverse
from .base_model import BaseModel

url = {
    'data-wavresnet': 'https://www.dropbox.com/s/q5bjbvm0tzdy4vk/epoch_best_n0124_loss0.00014028_psnr39.5489.pth?dl=1'
}

class WavResNet(BaseModel):
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
        parser.add_argument('--wavelet_func', type=str, default='haar', #'bior2.2',
            help='wavelet function ex: haar, bior2.2, or etc.')
        parser.add_argument('--lv', type=int, default=3,
            help='Level of stationary wavelet transform')
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
        self.var_name = ['x', 'out', 'target']

        # Create model
        self.net = create_model(opt).to(self.device)
        self.mse_loss_criterion = nn.MSELoss()
        
        # Define losses and optimizers
        if self.is_train:
            self.loss_criterion = nn.MSELoss()
            self.optimizer_names = ['optimizer']
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
            self.optimizers.append(self.optimizer)

        if opt.url:
            datasetname = ''
            for d in opt.datasets:
                datasetname = datasetname + d
            url_name = 'data-wavresnet'.format(datasetname)
            if url_name in url:
                self.url = url[url_name]
            else:
                print('url_name:', url_name)
                raise('Set model configurations correctly to load model from url')

    def set_input(self, input):
        self.x = input['x'].to(self.device)
        if input['target'] is not None:
            self.target = input['target'].to(self.device)

    def forward(self):
        self.out = self.net(self.x)

    def backward(self):
        self.loss = self.loss_criterion(self.out, self.target)
        mse_loss = self.mse_loss_criterion(self.out, self.target)
        self.psnr = 10 * torch.log10(1 / mse_loss)
        
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

# Create mwcnn model

def create_model(opt):
    return WavResNetModel(opt)

class WavBlock(nn.Module):
    def __init__(self, opt, inner_nc, outer_nc, norm_layer=nn.BatchNorm2d,use_dropout=False):
        super(WavBlock, self).__init__()
        # C = nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=True)
        # B = norm_layer(outer_nc)
        # R = nn.ReLU(True)

        # CBR = [C,B,R]
        # wav = CBR + CBR + CBR
        cbr1 = ConvBatchRelu(inner_nc, outer_nc, norm_layer=norm_layer)
        cbr2 = ConvBatchRelu(inner_nc, outer_nc, norm_layer=norm_layer)
        cbr3 = ConvBatchRelu(inner_nc, outer_nc, norm_layer=norm_layer)
        
        self.wavblock = nn.Sequential(*[cbr1, cbr2, cbr3])
        self.relu = nn.ReLU()

        # return nn.Sequential(*model)

    def forward(self, x):
        # out = x + self.cbr_block(x)
        out = x + self.wavblock(x)
        out = self.relu(out)
        return out

class ConvBatchRelu(nn.Module):
    def __init__(self, inner_nc, outer_nc, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ConvBatchRelu, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        C = nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
        B = norm_layer(outer_nc)
        R = nn.ReLU(True)

        CBR = [C, B, R]

        self.cbr_block = nn.Sequential(*CBR)

    def forward(self, x):
        out = self.cbr_block(x)
        return out

class WavResNetModel(nn.Module):
    def __init__(self, opt, ngf=128, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(WavResNetModel, self).__init__()

        self.lv = opt.lv
        self.nc = opt.n_channels

        input_nc = (3 * self.lv + 1) * self.nc
        output_nc = input_nc

        # inp_C = nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1, bias=True)
        # C = nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1, bias=True)
        # cat_C = nn.Conv2d(ngf*7, ngf, kernel_size=3, stride=1, padding=1, bias=True)
        # out_C = nn.Conv2d(ngf, output_nc, kernel_size=3, stride=1, padding=1, bias=True)

        B = norm_layer
        R = nn.ReLU(True)

        # inpModule = [inp_C,B,R] + [C,B,R] + [C,B,R]
        # outModule = [cat_C,B,R] + [C,B,R] + [out_C]
        inpModule = [
            ConvBatchRelu(input_nc, ngf, norm_layer=B),
            ConvBatchRelu(ngf, ngf, norm_layer=B),
            ConvBatchRelu(ngf, ngf, norm_layer=B)
        ]
        outModule = [
            ConvBatchRelu(ngf*7, ngf, norm_layer=B),
            ConvBatchRelu(ngf, ngf, norm_layer=B),
            nn.Conv2d(ngf, output_nc, kernel_size=3, stride=1, padding=1, bias=True)
        ]


        self.swt = SWTForward(J=self.lv, wave=opt.wavelet_func)
        self.inp_block = nn.Sequential(*inpModule)
        self.module1   = nn.Sequential(WavBlock(opt, inner_nc=ngf, outer_nc=ngf) , R)
        self.module2   = nn.Sequential(WavBlock(opt, inner_nc=ngf, outer_nc=ngf) , R)
        self.module3   = nn.Sequential(WavBlock(opt, inner_nc=ngf, outer_nc=ngf) , R)
        self.module4   = nn.Sequential(WavBlock(opt, inner_nc=ngf, outer_nc=ngf) , R)
        self.module5   = nn.Sequential(WavBlock(opt, inner_nc=ngf, outer_nc=ngf) , R)
        self.module6   = nn.Sequential(WavBlock(opt, inner_nc=ngf, outer_nc=ngf) , R)
        self.out_block = nn.Sequential(*outModule)
        self.iswt = SWTInverse(wave=opt.wavelet_func)

    def forward(self, x):
        x = self.swt(x)
        x = serialize_swt(x)
        reg0 = self.inp_block(x)
        reg1 = self.module1(reg0)
        reg2 = self.module2(reg1)
        reg3 = self.module3(reg2)
        reg4 = self.module4(reg3)
        reg5 = self.module5(reg4)
        reg6 = self.module6(reg5)
        cat_res = torch.cat([reg0,reg1,reg2,reg3,reg4,reg5,reg6],1)

        out = x + self.out_block(cat_res)
        out = unserialize_swt(out, J=self.lv, C=self.nc) # out = (ll, swt_coeffs)
        out = self.iswt(out)
        return out
