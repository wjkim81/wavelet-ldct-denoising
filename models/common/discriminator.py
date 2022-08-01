import torch
import torch.nn

def make_discriminator(opt):
    return Discriminator(opt.n_channels)
    

class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for out_filters, stride, normalize in [ (64, 1, False),
                                                (64, 2, True),
                                                (128, 1, True),
                                                (128, 2, True),
                                                (256, 1, True),
                                                (256, 2, True),
                                                (512, 1, True),
                                                (512, 2, True),]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        # Output layer
        #layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))
        out_layers = [
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        ]
        layers.extend(out_layers)

        self.model = nn.Sequential(*layers)
        """
        Check here
        https://github.com/twhui/SRGAN-pyTorch
        https://github.com/leftthomas/SRGAN/blob/master/model.py
        https://github.com/aitorzip/PyTorch-SRGAN
        """
