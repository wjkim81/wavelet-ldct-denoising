import torch
import torch.nn as nn
from torchvision.models import vgg19

def parse_perceptual_loss(parser):
    parser.add_argument('--content_loss', type=str, choices=['l1', 'l2'], default='l1',
        help='loss function (l1, l2)')
    parser.add_argument('--style_loss', type=str, choices=['l1', 'l2'], default='l1',
        help='loss function (l1, l2)')

    # parser.add_argument('--perceptual_loss', type=str, default=None,
    #     choices=['srgan', 'wavelet_transfer', 'perceptual_loss'],
    #     help='specity loss_type')
    parser.add_argument('--content_layers', nargs='+', default=[],
        choices=[
            'conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
            'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
            'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
            'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4'],
        help='specify content layers')
    parser.add_argument('--style_layers', nargs='+', default=[],
        choices=[
            'relu1_1', 'relu1_2', 'relu2_1', 'relu2_2',
            'relu3_1', 'relu3_2', 'relu3_3', 'relu3_4',
            'relu4_1', 'relu4_2', 'relu4_3', 'relu4_4',
            'relu5_1', 'relu5_2', 'relu5_3', 'relu5_4'],
        help='specify content layers')
    parser.add_argument('--gram_matrix', default=False, action='store_true',
        help='do Gram Matrix')
    parser.add_argument('--content_weights',  nargs='+', type=float, default=[],
        help='weight of content loss')
    parser.add_argument('--style_weights',  nargs='+', type=float, default=[],
        help='weight of style loss')

    return parser

def parse_wavelet_perceptual_loss(parser):
    parser.add_argument('--content_loss', type=str, choices=['l1', 'l2'], default='l1',
        help='loss function (l1, l2)')
    parser.add_argument('--style_loss', type=str, choices=['l1', 'l2'], default='l1',
        help='loss function (l1, l2)')

    parser.add_argument('--content_layers', nargs='+', default=[],
        choices=[
            'conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
            'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
            'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
            'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4'],
        help='specify content layers')
    parser.add_argument('--style_layers', nargs='+', default=[],
        choices=[
            'relu1_1', 'relu1_2', 'relu2_1', 'relu2_2',
            'relu3_1', 'relu3_2', 'relu3_3', 'relu3_4',
            'relu4_1', 'relu4_2', 'relu4_3', 'relu4_4',
            'relu5_1', 'relu5_2', 'relu5_3', 'relu5_4'],
        help='specify content layers')
    parser.add_argument('--gram_matrix', default=False, action='store_true',
        help='do Gram Matrix')
    parser.add_argument('--ll_content_weights',  nargs='+', type=float, default=[],
        help='weight of LL content loss')
    parser.add_argument('--ll_style_weights',  nargs='+', type=float, default=[],
        help='weight of LL style loss')
    parser.add_argument('--high_content_weights',  nargs='+', type=float, default=[],
        help='weight of high content loss')
    parser.add_argument('--high_style_weights',  nargs='+', type=float, default=[],
        help='weight of high style loss')

    return parser

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class VGG19(nn.Module):
    def __init__(self, content_layers, style_layers,
        content_loss_criterion, style_loss_criterion, style_gram=False):
        super(VGG19, self).__init__()

        cnn = vgg19(pretrained=True).features.eval()

        self.model = nn.Sequential()
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_loss_criterion = content_loss_criterion
        self.style_loss_criterion = style_loss_criterion
        self.style_gram = style_gram

        i = 1
        j = 0
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                j += 1
                name = 'conv{}_{}'.format(i, j)
            elif isinstance(layer, nn.ReLU):
                name = 'relu{}_{}'.format(i, j)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss. Some erroe occurs. So we replace with inplace=False
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                j = 0
                name = 'pool{}'.format(i)
                i += 1

            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn{}_{}'.format(i, j)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        
            self.model.add_module(name, layer)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        content_loss = []
        style_loss = []

        if x.size(1) == 1: x = torch.cat((x, x, x), 1)
        if y.size(1) == 1: y = torch.cat((y, y, y), 1)

        for (name, module) in self.model.named_children():
            x = module(x)
            y = module(y)
            if name in self.content_layers:
                loss = self.content_loss_criterion(x, y)
                content_loss.append(loss)
            elif name in self.style_layers:
                if self.style_gram:
                    x_f = gram_matrix(x)
                    y_f = gram_matrix(y)
                else:
                    x_f = x
                    y_f = y

                loss = self.style_loss_criterion(x_f, y_f)
                style_loss.append(loss)

        assert len(content_loss) == len(self.content_layers)
        assert len(style_loss) == len(self.style_layers)

        return (content_loss, style_loss)

class PerceptualLoss(nn.Module):
    def __init__(self, opt):
        super(PerceptualLoss, self).__init__()
        print('Use {} method for perceptual loss'.format(opt.perceptual_loss))
        if opt.perceptual_loss == 'srgan':
            content_layers = []
            style_layers = ['relu5_4']
            self.content_weights = [1.0]
            self.style_weights = [5e-3]
            gram_matrix=False
        elif opt.perceptual_loss == 'perceptual_loss':
            content_layers = []
            style_layers = ['relu2_2']
            self.content_weights = [0]
            self.style_weights = [1.0]
            gram_matrix = True
            # print('self.content_weights:', self.content_weights)
        elif opt.perceptual_loss == 'wavelet_transfer':
            content_layers = ['conv2_2']
            style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
            self.content_weights = [1.0]
            self.style_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
            gram_matrix = True
        elif opt.perceptual_loss == 'edsrganp':
            content_layers = []
            style_layers = ['relu5_4']
            self.content_weights = [0.1]
            self.style_weights = [1.0]
            gram_matrix=False
        elif opt.perceptual_loss == 'cpce3':
            content_layers = []
            style_layers = ['relu5_2']
            self.content_weights = [0]
            self.style_weights = [1.0]
            gram_matrix=False
        else:
            content_layers = opt.content_layers
            style_layers = opt.style_layers
            self.content_weights = opt.content_weights
            self.style_weights = opt.style_weights
            gram_matrix = opt.gram_matrix


        if len(content_layers) != 0:
            assert len(content_layers) == len(self.content_weights)
        if len(style_layers) != 0:
            assert len(style_layers) == len(self.style_weights)

        if opt.content_loss == 'l1':
            self.content_loss_criterion = nn.L1Loss()
        elif opt.content_loss == 'l2':
            self.content_loss_criterion = nn.MSELoss()
        
        if opt.style_loss == 'l1':
            self.style_loss_criterion = nn.L1Loss()
        elif opt.style_loss == 'l2':
            self.style_loss_criterion = nn.MSELoss()

        self.feature_extractor = VGG19(
            content_layers, style_layers,
            self.content_loss_criterion, self.style_loss_criterion,
            style_gram=gram_matrix
        ).to(opt.device)
        if opt.multi_gpu: self.feature_extractor = nn.DataParallel(self.feature_extractor)


    def forward(self, x, y):
        content_loss_list, style_loss_list = self.feature_extractor(x, y)
        # Calculate content loss
        if len(content_loss_list) == 0 and self.content_weights[0] == 0:
            content_loss = 0
        elif len(content_loss_list) == 0 and self.content_weights[0] != 0:
            content_loss = self.content_loss_criterion(x, y) * self.content_weights[0]
        else:
            content_loss = 0
            for (w, cl) in zip(self.content_weights, content_loss_list):
                cl = cl.mean()
                content_loss += w * cl

        # Calculate style loss
        if len(style_loss_list) == 0 and self.style_weights[0] == 0:
            style_loss = 0
        elif len(style_loss_list) == 0 and self.style_weights[0] != 0:
            raise RuntimeError("When perceptual loss is used, specify style_layers!")
        else:
            style_loss = 0
            for (w, sl) in zip(self.style_weights, style_loss_list):
                sl = sl.mean()
                style_loss += w * sl

        return (content_loss, style_loss)


class WaveletPerceptualLoss(nn.Module):
    def __init__(self, opt):
        super(WaveletPerceptualLoss, self).__init__()
        self.nc = opt.n_channels

        print('Use {} method for perceptual loss'.format(opt.perceptual_loss))
        if opt.perceptual_loss == 'srgan':
            content_layers = []
            style_layers = ['relu5_4']
            self.ll_content_weights = [1.0]
            self.ll_style_weights = [5e-3]
            self.high_content_weights = [0]
            self.high_style_weights = [5e-3]
            gram_matrix=False
        elif opt.perceptual_loss == 'perceptual_loss':
            content_layers = []
            style_layers = ['relu2_2']
            self.ll_content_weights = [1.0]
            self.ll_style_weights = [0.1]
            self.high_content_weights = [0]
            self.high_style_weights = [1.0]
            gram_matrix = True
        elif opt.perceptual_loss == 'wavelet_transfer':
            content_layers = ['conv2_2']
            style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
            self.ll_content_weights = [1.0]
            self.ll_style_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
            self.high_content_weights = [1.0]
            self.high_style_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
            gram_matrix = True
        elif opt.perceptual_loss == 'ganp1':
            content_layers = []
            style_layers = ['relu2_2']
            self.ll_content_weights = [1.0]
            self.ll_style_weights = [0.1]
            self.high_content_weights = [0.1]
            self.high_style_weights = [1.0]
            gram_matrix = True
        elif opt.perceptual_loss == 'ganp2':
            content_layers = []
            style_layers = ['relu2_2']
            self.ll_content_weights = [0.1]
            self.ll_style_weights = [1.0]
            self.high_content_weights = [0]
            self.high_style_weights = [1.0]
            gram_matrix = True
        elif opt.perceptual_loss == 'ganp3':
            content_layers = []
            style_layers = ['relu2_2']
            self.ll_content_weights = [1.0]
            self.ll_style_weights = [0.05]
            self.high_content_weights = [0.05]
            self.high_style_weights = [1.0]
            gram_matrix = True
        elif opt.perceptual_loss == 'ganp4':
            content_layers = []
            style_layers = ['relu2_2']
            self.ll_content_weights = [1.0]
            self.ll_style_weights = [0.5]
            self.high_content_weights = [0.05]
            self.high_style_weights = [1.0]
            gram_matrix = True
        elif opt.perceptual_loss == 'ganp5':
            content_layers = []
            style_layers = ['relu2_2']
            self.ll_content_weights = [0.1]
            self.ll_style_weights = [1.0]
            self.high_content_weights = [0.05]
            self.high_style_weights = [1.0]
            gram_matrix = True
        elif opt.perceptual_loss == 'ganp6':
            content_layers = []
            style_layers = ['relu2_2']
            self.ll_content_weights = [0.01]
            self.ll_style_weights = [1.0]
            self.high_content_weights = [0.01]
            self.high_style_weights = [1.0]
            gram_matrix = True
        elif opt.perceptual_loss == 'ganp7':
            content_layers = []
            style_layers = ['relu2_2']
            self.ll_content_weights = [0.01]
            self.ll_style_weights = [1.0]
            self.high_content_weights = [0.01]
            self.high_style_weights = [1.0]
            gram_matrix = False
        elif opt.perceptual_loss == 'ganp8':
            content_layers = []
            style_layers = ['relu2_2']
            self.ll_content_weights = [1.0]
            self.ll_style_weights = [0.5]
            self.high_content_weights = [0]
            self.high_style_weights = [1.0]
            gram_matrix = True
        else:
            content_layers = opt.content_layers
            style_layers = opt.style_layers
            self.ll_content_weights = opt.ll_content_weights
            self.ll_style_weights = opt.ll_style_weights
            self.high_content_weights = opt.high_content_weights
            self.high_style_weights = opt.high_style_weights
            gram_matrix = opt.gram_matrix

        if len(content_layers) != 0:
            assert len(content_layers) == len(self.ll_content_weights)
            assert len(content_layers) == len(self.high_content_weights)
        if len(style_layers) != 0:
            assert len(style_layers) == len(self.ll_style_weights)
            assert len(style_layers) == len(self.high_style_weights)

        if opt.content_loss == 'l1':
            self.content_loss_criterion = nn.L1Loss()
        elif opt.content_loss == 'l2':
            self.content_loss_criterion = nn.MSELoss()
        
        if opt.style_loss == 'l1':
            self.style_loss_criterion = nn.L1Loss()
        elif opt.style_loss == 'l2':
            self.style_loss_criterion = nn.MSELoss()

        self.feature_extractor = VGG19(
            content_layers, style_layers,
            self.content_loss_criterion, self.style_loss_criterion,
            style_gram=gram_matrix
        ).to(opt.device)
        if opt.multi_gpu: self.feature_extractor = nn.DataParallel(self.feature_extractor)

    def forward(self, swt_x, swt_y):

        ll_x = swt_x[:, :self.nc]
        ll_y = swt_y[:, :self.nc]
        high_x = swt_x[:, self.nc:]
        high_y = swt_y[:, self.nc:]


        # Calculate LL content loss
        content_loss_list, style_loss_list = self.feature_extractor(ll_x, ll_y)
        if len(content_loss_list) == 0 and self.ll_content_weights[0] == 0:
            ll_content_loss = 0
        elif len(content_loss_list) == 0 and self.ll_content_weights[0] != 0:
            ll_content_loss = self.content_loss_criterion(ll_x, ll_y) * self.ll_content_weights[0]
        else:
            ll_content_loss = 0
            for (w, cl) in zip(self.ll_content_weights, content_loss_list):
                cl = cl.mean()
                ll_content_loss += w * cl

        # Calculate LL style loss
        if len(style_loss_list) == 0 and self.ll_style_weights[0] == 0:
            ll_style_loss = 0
        elif len(style_loss_list) == 0 and self.ll_style_weights[0] != 0:
            raise RuntimeError("When perceptual loss is used, specify style_layers!")
        else:
            ll_style_loss = 0
            for (w, sl) in zip(self.ll_style_weights, style_loss_list):
                sl = sl.mean()
                ll_style_loss += w * sl

        # Calculate hight content loss
        num_swt_ch = high_x.size(1)
        high_content_loss = 0
        high_style_loss = 0
        for c in range(0, num_swt_ch, self.nc):
            sb_x = high_x[:, c:c+self.nc]
            sb_y = high_y[:, c:c+self.nc]
            content_loss_list, style_loss_list = self.feature_extractor(sb_x, sb_y)
            
            if len(content_loss_list) == 0 and self.high_content_weights[0] == 0:
                sb_cl = 0
            elif len(content_loss_list) == 0 and self.high_content_weights[0] != 0:
                sb_cl = self.content_loss_criterion(sb_x, sb_y) * self.high_content_weights[0]
            else:
                sb_cl = 0
                for (w, cl) in zip(self.high_content_weights, content_loss_list):
                    cl = cl.mean()
                    sb_cl += w * cl
            high_content_loss += sb_cl

            if len(style_loss_list) == 0 and self.high_style_weights[0] == 0:
                sb_sl = 0
            elif len(style_loss_list) == 0 and self.high_style_weights[0] != 0:
                raise RuntimeError("When perceptual loss is used, specify style_layers!")
            else:
                sb_sl = 0
                for (w, sl) in zip(self.high_style_weights, style_loss_list):
                    sl = sl.mean()
                    sb_sl += w * sl
            high_style_loss += sb_sl

        return ll_content_loss, ll_style_loss, high_content_loss, high_style_loss

