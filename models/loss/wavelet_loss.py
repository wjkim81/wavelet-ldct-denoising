"""
Perceptual Losses for Real-Time Style Transfer and Super-Resolution
https://arxiv.org/abs/1603.08155

Refer the following tutorial to calculate style loss
https://pytorch.org/tutorials/advanced/neural_style_tutorial.html#loading-the-images
"""
import torch
import torch.nn as nn
from torchvision.models import vgg19


def parse_perceptual_loss(parser):
    parser.add_argument('--content_loss', type=str, choices=['l1', 'l2'],
        help='loss function (l1, l2)')
    parser.add_argument('--style_loss', type=str, choices=['l1', 'l2'],
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

class Normalization(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

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

                # print("name:", name)
                # print("x_f.shape:", x_f.shape)
                # print("y_f.shape:", y_f.shape)
                # print("self.style_loss_criterion:", self.style_loss_criterion)
                loss = self.style_loss_criterion(x_f, y_f)
                style_loss.append(loss)

        assert len(content_loss) == len(self.content_layers)
        assert len(style_loss) == len(self.style_layers)

        return (content_loss, style_loss)

def get_content_loss(input_features, target_features, loss_criterion, content_weights):
    total_content_loss = 0.
    for (w, input_f, target_f) in zip(content_weights, input_features, target_features):
        # print("content w:", w)
        # print("input_f.shape:", input_f.shape)
        # print("target_f.shape:", target_f.shape)
        if w != 0:
            total_content_loss += w * loss_criterion(input_f, target_f)

    # print("total_content_loss:", total_content_loss)
    return total_content_loss
    
def get_style_loss(input_features, target_features, loss_criterion, style_weights, gram=False):
    total_style_loss = 0.
    for (w, input_f, target_f) in zip(style_weights, input_features, target_features):

        # print("style w:", w)
        # print("input_f.shape:", input_f.shape)
        # print("target_f.shape:", target_f.shape)
        if w != 0:
            if gram:
                input_f = gram_matrix(input_f)
                target_f = gram_matrix(target_f)

            # style_loss = loss_criterion(input_f, target_f)
            # print("style_loss:", style_loss)

            total_style_loss += w * loss_criterion(input_f, target_f)

    # print("total_style_loss:", total_style_loss)
    return total_style_loss

def get_high_content_style_loss(
    feature_extractor, input_img, target_img, content_weights, style_weights, content_loss_criterion, n_channels):

    content_loss = 0 # torch.tensor([0], dtype=torch.float)
    style_loss = 0 # torch.tensor([0], dtype=torch.float)

    # for c in range(num_channels - 1):
    num_swt_ch = input_img.size(1)
    for c in range(0, num_swt_ch, n_channels):
        input_c = input_img[:,c:c+n_channels,:,:]
        target_c = target_img[:,c:c+n_channels,:,:]
        content_loss_list, style_loss_list = feature_extractor(input_c, target_c)

        if len(content_loss_list) == 0 and content_weights[0] == 0:
            content_loss = 0
        elif len(content_loss_list) == 0 and content_weights[0] != 0:
            content_loss = content_loss_criterion(target_c, input_c) * content_weights[0]
            # print("cl[{}]: {:5f}, cw: {}".format(c+1, content_loss, content_weights[0]), end=", ")
        else:
            for (w, cl) in zip(content_weights, content_loss_list):
                cl = cl.mean()
                # print("cl[{}]: {:5f}, cw: {}".format(c+1, content_loss, content_weights[0]), end=", ")
                content_loss += w * cl

        if len(style_loss_list) == 0 and style_weights[0] == 0:
            style_loss = 0
        elif len(style_loss_list) == 0 and style_weights[0] != 0:
            raise RuntimeError("When perceptual loss is used, specify style_layers!")
        else:
            for (w, sl) in zip(style_weights, style_loss_list):
                sl = sl.mean()
                # print("sl[{}]: {:5f}, sw: {}".format(c+1, sl, w), end=" ")
                style_loss += w * sl
            # print("")

    return (content_loss, style_loss)

def get_ll_content_style_loss(
    feature_extractor, input_img, target_img, ll_content_weights, ll_style_weights, content_loss_criterion):

    ll_content_loss = 0 # torch.tensor([0], dtype=torch.float)
    ll_style_loss = 0 # torch.tensor([0], dtype=torch.float)

    ll_content_loss_list, ll_style_loss_list = feature_extractor(input_img, target_img)

    if len(ll_content_loss_list) == 0 and ll_content_weights[0] == 0:
        ll_content_loss = 0
    elif len(ll_content_loss_list) == 0 and ll_content_weights[0] != 0:
        ll_content_loss = content_loss_criterion(target_img, input_img) * ll_content_weights[0]
    else:
        for (w, cl) in zip(ll_content_weights, ll_content_loss_list):
            cl = cl.mean()
            ll_content_loss += w * cl

    if len(ll_style_loss_list) == 0 and ll_style_weights[0] == 0:
        ll_style_loss = 0
    elif len(ll_style_loss_list) == 0 and ll_style_weights[0] != 0:
        raise RuntimeError("When perceptual loss is used, specify style_layers!")
    else:
        for (w, sl) in zip(ll_style_weights, ll_style_loss_list):
            sl = sl.mean()
            ll_style_loss += w * sl

    return (ll_content_loss, ll_style_loss)

class CriterionPerceptual:
    def __init__(
        self, feature_extractor,
        ll_content_weights, ll_style_weights, high_content_weights, high_style_weights, 
        content_loss_criterion, n_channels):
    
        self.feature_extractor = feature_extractor
        self.ll_content_weights = ll_content_weights
        self.ll_style_weights = ll_style_weights
        self.high_content_weights = high_content_weights
        self.high_style_weights = high_style_weights
        self.content_loss_criterion = content_loss_criterion
        self.nc = n_channels

    def get_ll_loss(self, target_f, input_f):
        return get_ll_content_style_loss(self.feature_extractor, target_f, input_f,
                    self.ll_content_weights, self.ll_style_weights, self.content_loss_criterion)
    
    def get_high_loss(self, target_f, input_f):
        return get_high_content_style_loss(self.feature_extractor, target_f, input_f,
                    self.high_content_weights, self.high_style_weights, self.content_loss_criterion, self.nc)


