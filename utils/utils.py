import os
from collections import OrderedDict
import numpy as np
import torch.nn as nn

from models.convs.wavelet import SWTForward, SWTInverse

def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

def print_subband_loss(opt, spectrum_txt, target, out, swt):
    if opt.content_loss == 'l1':
        loss_criterion = nn.L1Loss()
    elif opt.content_loss == 'l2':
        loss_criterion = nn.MSELoss()
    
    swt_target = swt(target)
    swt_out = swt(out)

    spectrum_path = os.path.join(opt.savedir, spectrum_txt)
    loss_dict = OrderedDict()

    J = len(swt_target)
    for j in range(J):
        target_coeff = swt_target[j]
        out_coeff = swt_out[j]

        for s in range(target_coeff.size(2)):
            target_sb = target_coeff[:, :, s]
            out_sb = out_coeff[:, :, s]

            sb_loss = loss_criterion(target_sb, out_sb)
            loss_dict[str(j) + '-' + str(s)] = sb_loss

    hline = ''
    line = ''
    for key, value in loss_dict.items():
        hline += "{},".format(key)
        line += "{:.8f},".format(value)
        # f.write("{:.8f},".format(value))

    line = line[:-1] + '\n'

    if not os.path.exists(spectrum_path):
        hline = hline[:-1] + '\n'
        with open(spectrum_path, 'w') as f:
            f.write(hline)

    with open(spectrum_path, 'a') as f:
        f.write(line)
