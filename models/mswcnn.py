import torch
import torch.nn as nn

from models.loss.perceptual_loss import parse_wavelet_perceptual_loss, WaveletPerceptualLoss
from models.convs.wavelet import SWTForward, SWTInverse, serialize_swt, itransformer
from models.convs import common
from models.convs.norm import Norm2d, Norm2d_
from .base_model import BaseModel

# url = {
#     'data-mayof32g32b8l16': 'https://www.dropbox.com/s/q5bjbvm0tzdy4vk/epoch_best_n0124_loss0.00014028_psnr39.5489.pth?dl=1'
# }

class MSWCNN(BaseModel):
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
        parser.add_argument('--wavelet', type=str, default='haar', #'bior2.2',
            help='wavelet function ex: haar, bior2.2, or etc.')
        parser.add_argument('--swt_lv', type=int, default=2,
            help='Level of stationary wavelet transform')
        

        # Network parameters
        parser.add_argument('--n_feats', type=int, default=32,
            help='number of features')
        parser.add_argument('--growth_rate', type=int, default=16,
            help='growth rate of each layer in dense block')
        parser.add_argument('--n_blocks', type=int, default=16,
            help='number of dense blocks')
        parser.add_argument('--n_layers', type=int, default=8,
            help='number of layers of dense blocks')


        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout

        parser.add_argument('--perceptual_loss', type=str, default=None,
            choices=['srgan', 'wavelet_transfer', 'perceptual_loss'],
            help='specity loss_type')
        if is_train:
            # Loss
            parser = parse_wavelet_perceptual_loss(parser)
            parser.add_argument('--img_loss', default=False, action='store_true',
                help='include img loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.model_names = ['net']
        self.loss_name = ['loss']
        self.var_name = ['x', 'out', 'target']

        # Create model
        self.net = create_model(opt).to(self.device)
        self.mse_loss_criterion = nn.MSELoss()
        
        # Define losses and optimizers
        if self.is_train:
            self.loss_criterion = nn.L1Loss()
            self.optimizer_names = ['optimizer']
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
            self.optimizers.append(self.optimizer)

        if opt.url:
            url_name = 'data-{}f{}g{}b{}l{}'.format(self._name_dataset(opt), opt.n_feats, opt.growth_rate, opt.n_layers, opt.n_blocks)
            if url_name in url:
                self.url_name = url_name
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
        # print('forwarding')
        self.forward()
        # print('backward')
        self.backward()
        # print('step')
        self.optimizer.step()
    
    def log_loss(self, opt, phase, batch_time, iter, n_iter):
        print("{} {:.3f}s => Epoch[{}/{}]({}/{}): Loss: {:.5f}, PSNR: {:.5f}".format(
            phase, batch_time, opt.epoch, opt.n_epochs, iter, n_iter,
            self.loss.item(), self.psnr.item())
        )

# Create mwcnn model

def create_model(opt):
    return MSWCNNModel(opt)


"""
Define MSWCNN model
"""

class SWT(nn.Module):
    def __init__(self, wave):
        super(SWT, self).__init__()
        self.swt = SWTForward(J=1, wave=wave)

    def forward(self, x):
        x = self.swt(x)
        x = serialize_swt(x)
        return x


class ISWT(nn.Module):
    def __init__(self, wave, n_channels):
        super(ISWT, self).__init__()
        self.n_channels = n_channels
        self.iswt = SWTInverse(wave=wave)

    def forward(self, x):
        ll, coeffs = itransformer(x)
        x = self.iswt((ll, coeffs))
        return x

class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class SWTRDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, swt):
        super(SWTRDB, self).__init__()
        self.swt = swt
        swt_channels = in_channels * 4
        self.layers = nn.Sequential(*[DenseLayer(swt_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(swt_channels + growth_rate * num_layers, in_channels, kernel_size=1)

    def forward(self, x):
        # print('x.shape:', x.shape)
        swt_x = self.swt(x)
        # print('swt_x.shape:', swt_x.shape)
        
        # print('self.layers(x).shape:', self.layers(swt_x).shape)
        # print('self.lff(self.layers(x)).shape', self.lff(self.layers(swt_x)).shape)
        
        return x + self.lff(self.layers(swt_x))  # local residual learning


class ISWTBlock(nn.Module):
    def __init__(self, conv, iswt, in_feats, out_feats, kernel_size,
        n_resblocks=3, bias=True, bn=False, act=nn.ReLU(True), lv=1):
        super(ISWTBlock, self).__init__()

        self.iswt = iswt

        assert in_feats // 4 == out_feats
        m = [
            common.ResBlock(
                conv, out_feats, kernel_size, bias=bias, bn=bn, act=act
            ) for _ in range(n_resblocks)
        ]

        self.body = nn.Sequential(*m)

    def forward(self, x):
        # print('x.shape:', x.shape)
        iswtx = self.iswt(x)
        # print('iswt.shape:', iswtx.shape)
        res = iswtx
        x = self.body(iswtx)
        
        out = res + x
        return out

class MSWCNNModel(nn.Module):
    def __init__(self, opt):
        super(MSWCNNModel, self).__init__()

        num_channels = opt.n_channels
        num_features = opt.n_feats
        growth_rate = opt.growth_rate
        num_blocks = opt.n_blocks
        num_layers = opt.n_layers

        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        wave = opt.wavelet
        swt = SWT(wave)
        iswt = ISWT(wave, num_channels)


        self.sub_mean = common.MeanShift(1.0, n_channels=num_channels)
        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.swt_rdbs = nn.ModuleList([SWTRDB(self.G0, self.G, self.C, swt)])
        for _ in range(self.D - 1):
            self.swt_rdbs.append(SWTRDB(self.G0, self.G, self.C, swt))

        # global feature fusion
        # self.gff = nn.Sequential(
        #     nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
        #     nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        # )

        self.gff = nn.Conv2d(self.G0 * self.D, self.G0 * self.D, kernel_size=1)

        self.iswt_blocks = nn.Sequential(*[
            ISWTBlock(common.default_conv, iswt, self.G0 * self.D, (self.G0 * self.D) // 4, 3),
            ISWTBlock(common.default_conv, iswt, (self.G0 * self.D) // 4, self.G0, 3)
        ])

        self.output = nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)

        self.add_mean = common.MeanShift(1.0, n_channels=num_channels, sign=1)


    def forward(self, x, update_stat=False):
        x = self.sub_mean(x)
        global_res = x
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.swt_rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1))  # global residual learning
        # print('feature fusion:', x.shape)
        x = self.iswt_blocks(x) # + sfe1
        x = x + sfe1
        x = self.output(x) #+ global_res
        x = x + global_res

        x = self.add_mean(x)
        return x
