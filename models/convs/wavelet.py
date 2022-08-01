"""
https://github.com/fbcotter/pytorch_wavelets
https://github.com/fbcotter/pytorch_wavelets/blob/master/pytorch_wavelets/dwt/transform2d.py
https://github.com/fbcotter/pytorch_wavelets/blob/master/pytorch_wavelets/dwt/swt_inverse.py

If you have a problem with this code, test with tests/swt.py file.
"""
import torch
import torch.nn as nn
import numpy as np

import pywt
import pytorch_wavelets.dwt.lowlevel as lowlevel
import torch.nn.functional as F


def standarize_coeffs(coeffs, ch_mean, ch_std):
    if not ch_mean or not ch_std:
        raise ValueError("ch_mean or ch_std is empty")
    out_coeffs = coeffs.clone()
    for i, (mean, std) in enumerate(zip(ch_mean, ch_std)):
        out_coeffs[:,i:i+1,:,:] = (coeffs[:,i:i+1,:,:] - mean) / std
    return out_coeffs

def unstandarize_coeffs(coeffs, ch_mean, ch_std):
    if not ch_mean or not ch_std:
        raise ValueError("ch_mean or ch_std is empty")
    out_coeffs = coeffs.clone()
    for i, (mean, std) in enumerate(zip(ch_mean, ch_std)):
        out_coeffs[:,i:i+1,:,:] = coeffs[:,i:i+1,:,:]  * std + mean
    return out_coeffs

def sfb1d(lo, hi, g0, g1, j, mode='periodic', dim=-1):
    """ 1D synthesis filter bank of an image tensor
    """
    C = lo.shape[1]
    d = dim % 4
    # If g0, g1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(g0, torch.Tensor):
        g0 = torch.tensor(np.copy(np.array(g0).ravel()),
                          dtype=torch.float, device=lo.device)
    if not isinstance(g1, torch.Tensor):
        g1 = torch.tensor(np.copy(np.array(g1).ravel()),
                          dtype=torch.float, device=lo.device)
    L = g0.numel()
    shape = [1,1,1,1]
    shape[d] = L
    N = 2*lo.shape[d]
    # If g aren't in the right shape, make them so
    if g0.shape != tuple(shape):
        g0 = g0.reshape(*shape)
    if g1.shape != tuple(shape):
        g1 = g1.reshape(*shape)

    s = (2, 1) if d == 2 else (1, 2)
    g0 = torch.cat([g0]*C,dim=0)
    g1 = torch.cat([g1]*C,dim=0)
    if mode == 'per' or mode == 'periodization':
        y = F.conv_transpose2d(lo, g0, stride=s, groups=C) + \
            F.conv_transpose2d(hi, g1, stride=s, groups=C)
        if d == 2:
            y[:,:,:L-2] = y[:,:,:L-2] + y[:,:,N:N+L-2]
            y = y[:,:,:N]
        else:
            y[:,:,:,:L-2] = y[:,:,:,:L-2] + y[:,:,:,N:N+L-2]
            y = y[:,:,:,:N]
        y = roll(y, 1-L//2, dim=dim)
    elif mode == 'zero' or mode == 'symmetric' or mode == 'reflect' or \
        mode == 'periodic':

        pp = L // 2
        # mypad = (0, 0, a, b) if d == 2 else (a, b, 0, 0)
        period_pad = (0, 0, pp, pp) if d == 2 else (pp, pp, 0, 0)
        p = 2 * pp - 1  + L // 2
        pad = (p, 0) if d == 2 else (0, p)

        lo = lowlevel.mypad(lo, pad=period_pad, mode=mode)
        hi = lowlevel.mypad(hi, pad=period_pad, mode=mode)

        y = F.conv_transpose2d(lo, g0, stride=s, padding=pad, groups=C) + \
            F.conv_transpose2d(hi, g1, stride=s, padding=pad, groups=C)
    else:
        raise ValueError("Unkown pad type: {}".format(mode))

    return y

def sfb2d(ll, lh, hl, hh, filts, j, mode='periodic'):
    """ Does a single level 2d wavelet reconstruction of wavelet coefficients.
    Does separate row and column filtering by two calls to
    :py:func:`pytorch_wavelets.dwt.lowlevel.sfb1d`
    Inputs:
        ll (torch.Tensor): lowpass coefficients
        lh (torch.Tensor): horizontal coefficients
        hl (torch.Tensor): vertical coefficients
        hh (torch.Tensor): diagonal coefficients
        filts (list of ndarray or torch.Tensor): If a list of tensors has been
            given, this function assumes they are in the right form (the form
            returned by
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_sfb2d`).
            Otherwise, this function will prepare the filters to be of the right
            form by calling
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_sfb2d`.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. Which
            padding to use. If periodization, the output size will be half the
            input size.  Otherwise, the output size will be slightly larger than
            half.
    """
    tensorize = [not isinstance(x, torch.Tensor) for x in filts]
    if len(filts) == 2:
        g0, g1 = filts
        if True in tensorize:
            g0_col, g1_col, g0_row, g1_row = lowlevel.prep_filt_sfb2d(g0, g1)
        else:
            g0_col = g0
            g0_row = g0.transpose(2,3)
            g1_col = g1
            g1_row = g1.transpose(2,3)
    elif len(filts) == 4:
        if True in tensorize:
            g0_col, g1_col, g0_row, g1_row = lowlevel.prep_filt_sfb2d(*filts)
        else:
            g0_col, g1_col, g0_row, g1_row = filts
    else:
        raise ValueError("Unknown form for input filts")

    lo = sfb1d(ll, lh, g0_col, g1_col, j, mode=mode, dim=2)
    hi = sfb1d(hl, hh, g0_col, g1_col, j, mode=mode, dim=2)
    y = sfb1d(lo, hi, g0_row, g1_row, j, mode=mode, dim=3)

    return y

def serialize_swt(swt_coeffs):
    """
    Iput:
        swt_coeffs:
            List of coefficients for each scale. Each coefficient has
            shape :math:`(N, C_{in}, 4, H_{in}, W_{in})` where the extra
            dimension stores the 4 subbands for each scale. The ordering in
            these 4 coefficients is: (A, H, V, D) or (ll, lh, hl, hh).
    Output:
       serialized coeffs of torch tensor (N, 4 * C_{in}, H_{in}, W_{in})
       Ex:
           LL_{j-1, 0}, LL_{j-1, 1}, ..., LL_{j-1, c}
           LH_{j-1, 0}, LH_{j-1, 1}, ..., LH_{j-1, c}
           HL_{j-1, 0}, HL_{j-1, 1}, ..., HL_{j-1, c}
           HH_{j-1, 0}, HH_{j-1, 1}, ..., HH_{j-1, c}
           ...
           no LL except for (j-1) level
           LH_{0, 0}, LH_{0, 1}, ..., LH_{0, c}
           HL_{0, 0}, HL_{0, 1}, ..., HL_{0, c}
           HH_{0, 0}, HH_{0, 1}, ..., HH_{0, c}
           
    """
    J = len(swt_coeffs)
    swt_coeffs_l = []
    for j in reversed(range(J)):
        if j == J - 1:
            coeffs = swt_coeffs[j][:, :, :]
        else:
            coeffs = swt_coeffs[j][:, :, 1:]
        bs, nc, sb, h, w = coeffs.shape
        # coeffs = coeffs.reshape(bs, c * sb, h, w)

        for s in range(sb):
            for c in range(nc):
                i = c + s * nc
                swt_coeffs_l.append(coeffs[:, c:c+1, s])

        # swt_coeffs_l.append(coeffs)

    swt_coeffs_l = torch.cat(swt_coeffs_l, 1)

    return swt_coeffs_l

def unserialize_swt(swt_coeffs_l, J, C):
    """
        Input:
            swt_coeffs: serialized swt coefficients
            J: swt level
            C: number of channel
        Output:
            Swt coefficients with output form of SWTForward
    """
    swt_coeffs = []
    for j in reversed(range(J)):
        if j == 0:
            ll= swt_coeffs_l[:, :C]
            coeffs = swt_coeffs_l[:, C:4*C]
        else:
            start_idx = 4 * C + (j - 1)*(3 * C) 
            coeffs = swt_coeffs_l[:, start_idx:start_idx+3*C]
        
        bs, ns, h, w = coeffs.shape
        tmp_coeffs = torch.zeros((bs, C, 3, h, w), device=coeffs.device)

        for s in range(3):
            for c in range(C):
                i = c + s * C
                tmp_coeffs[:, c, s] = coeffs[:, i]

        swt_coeffs.append(tmp_coeffs)

    return ll, swt_coeffs

 
def transformer(DMT1_yl, DMT1_yh):
    list_tensor = []
    for i in range(3):
        list_tensor.append(DMT1_yh[0][:,:,i,:,:])
    list_tensor.append(DMT1_yl)
    return torch.cat(list_tensor, 1)

def itransformer(out):
    yh = []
    C = int(out.shape[1]/4)
    # print(out.shape[0])
    y = out.reshape((out.shape[0], C, 4, out.shape[-2], out.shape[-1]))
    yl = y[:,:,0].contiguous()
    yh.append(y[:,:,1:].contiguous())

    return yl, yh

def swt_dict1(swt_coeffs):
    J = len(swt_coeffs)
    swt_coeffs_dict = {}
    for j in range(J):
        coeffs = swt_coeffs[j]
        bs, nc, sb, h, w = coeffs.shape
        if j == J - 1:
            s_range = range(0, sb)
        else:
            s_range = range(1, sb)
        for s in s_range:
            for c in range(nc):
                if s == 0:
                    subband = 'LL'
                elif s == 1:
                    subband = 'LH'
                elif s == 2:
                    subband = 'HL'
                else:
                    subband = 'HH'
                
                swt_coeffs_dict['{}{}_C{}'.format(subband, j+1, c)] = coeffs[:, c:c+1, s]
            
        if j == J - 1:
            coeffs = swt_coeffs[j][:, :, :]
        else:
            coeffs = swt_coeffs[j][:, :, 1:]

    return swt_coeffs_dict

def swt_dict2(swt_coeffs_l, J, C):
    swt_coeffs_dict = {}
    for j in range(J):
        start_idx = j*(C*3)
        if j == J - 1:
            for s in range(4):
                for c in range(C):
                    if s == 0:
                        subband = 'LL'
                    elif s == 1:
                        subband = 'LH'
                    elif s == 2:
                        subband = 'HL'
                    else:
                        subband == 'HH'
                    swt_coeffs_dict['{}{}_C{}'.format(subband, J+1, c)] = swt_coeffs_l[start_idx+4*c+s]
        else:
            for s in range(3):
                for c in range(C):
                    if s == 0:
                        subband = 'LH'
                    elif s == 1:
                        subband = 'HL'
                    else:
                        subband == 'HH'
                    swt_coeffs_dict['{}{}_C{}'.format(subband, J+1, c)] = swt_coeffs_l[start_idx+3*c+s]

    return swt_coeffs_dict


class SWTForward(nn.Module):
    """ Performs a 2d Stationary wavelet transform (or undecimated wavelet
    transform) of an image
    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet): Which wavelet to use. Can be a string to
            pass to pywt.Wavelet constructor, can also be a pywt.Wavelet class,
            or can be a two tuple of array-like objects for the analysis low and
            high pass filters.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme. PyWavelets uses only periodization so we use this
            as our default scheme.
        """
    def __init__(self, J=1, wave='db1', mode='periodic'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
        else:
            if len(wave) == 2:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = h0_col, h1_col
            elif len(wave) == 4:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = wave[2], wave[3]

        # Prepare the filters
        filts = lowlevel.prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row)
        self.h0_col = nn.Parameter(filts[0], requires_grad=False)
        self.h1_col = nn.Parameter(filts[1], requires_grad=False)
        self.h0_row = nn.Parameter(filts[2], requires_grad=False)
        self.h1_row = nn.Parameter(filts[3], requires_grad=False)

        self.J = J
        self.mode = mode

    def forward(self, x):
        """ Forward pass of the SWT.
        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`
        Returns:
            List of coefficients for each scale. Each coefficient has
            shape :math:`(N, C_{in}, 4, H_{in}, W_{in})` where the extra
            dimension stores the 4 subbands for each scale. The ordering in
            these 4 coefficients is: (A, H, V, D) or (ll, lh, hl, hh).
        """
        c = x.size(1)
        ll = x
        coeffs = []
        # Do a multilevel transform
        filts = (self.h0_col, self.h1_col, self.h0_row, self.h1_row)
        for j in range(self.J):
            # Do 1 level of the transform
            y = lowlevel.afb2d_atrous(ll, filts, self.mode, 2**j)

            bs, ch, h, w = y.shape
            y = y.view((bs, c, 4, h, w))

            coeffs.append(y)
            l = []
            for i in range(c): l.append(y[:, i, :1, :, :])
            
            # ll = y[:, :c, :, :]
            ll = torch.cat(l, 1)

        return coeffs

class SWTInverse(nn.Module):
    """ Performs a 2d DWT Inverse reconstruction of an image
    Args:
        wave (str or pywt.Wavelet): Which wavelet to use
        C: deprecated, will be removed in future
    """
    def __init__(self, wave='db1', mode='periodic', separable=True):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0_col, g1_col = wave.rec_lo, wave.rec_hi
            g0_row, g1_row = g0_col, g1_col
        else:
            if len(wave) == 2:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = g0_col, g1_col
            elif len(wave) == 4:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = wave[2], wave[3]
        # Prepare the filters
        if separable:
            filts = lowlevel.prep_filt_sfb2d(g0_col, g1_col, g0_row, g1_row)
            self.g0_col = nn.Parameter(filts[0], requires_grad=False)
            self.g1_col = nn.Parameter(filts[1], requires_grad=False)
            self.g0_row = nn.Parameter(filts[2], requires_grad=False)
            self.g1_row = nn.Parameter(filts[3], requires_grad=False)
        else:
            filts = lowlevel.prep_filt_sfb2d_nonsep(
                g0_col, g1_col, g0_row, g1_row)
            self.h = nn.Parameter(filts, requires_grad=False)
        self.mode = mode
        self.separable = separable

    def forward(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
              yl is a lowpass tensor of shape :math:`(N, C_{in}, H_{in}',
              W_{in}')` and yh is a list of bandpass tensors of shape
              :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. I.e. should match
              the format returned by DWTForward
        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, H_{in}, W_{in})`
        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        Note:
            Can have None for any of the highpass scales and will treat the
            values as zeros (not in an efficient way though).
        """
        yl, yh = coeffs
        ll = yl
        num_levels = len(yh)

        # out = torch.zeros(yl.shape, dtype=yl.dtype)
        out = ll.clone()

        # Do a multilevel inverse transform
        for j in range(num_levels):
            step_size = int(pow(2, num_levels-j-1))
            last_index = step_size

            h = yh[num_levels - 1 - j]
            if h is None:
                h = torch.zeros(ll.shape[0], ll.shape[1], 3, ll.shape[-2],
                    ll.shape[-1], device=ll.device)
            lh, hl, hh = torch.unbind(h, dim=2)
            filts = (self.g0_col, self.g1_col, self.g0_row, self.g1_row)
            
            for first_h in range(last_index):
                for first_w in range(last_index):
                    indices_h = slice(first_h, ll.shape[2], step_size)
                    indices_w = slice(first_w, ll.shape[3], step_size)

                    even_idx_h = slice(first_h, ll.shape[2], 2*step_size)
                    even_idx_w = slice(first_w, ll.shape[3], 2*step_size)

                    odd_idx_h = slice(first_h + step_size, ll.shape[2], 2*step_size)
                    odd_idx_w = slice(first_w + step_size, ll.shape[3], 2*step_size)

                    x1 = sfb2d(
                        out[..., even_idx_h, even_idx_w],
                        lh[..., even_idx_h, even_idx_w],
                        hl[..., even_idx_h, even_idx_w],
                        hh[..., even_idx_h, even_idx_w],
                        filts, j, mode=self.mode
                    )

                    x2 = sfb2d(
                        out[..., even_idx_h, odd_idx_w],
                        lh[..., even_idx_h, odd_idx_w],
                        hl[..., even_idx_h, odd_idx_w],
                        hh[..., even_idx_h, odd_idx_w],
                        filts, j, mode=self.mode
                    )

                    x3 = sfb2d(
                        out[..., odd_idx_h, even_idx_w],
                        lh[..., odd_idx_h, even_idx_w],
                        hl[..., odd_idx_h, even_idx_w],
                        hh[..., odd_idx_h, even_idx_w],
                        filts, j, mode=self.mode
                    )

                    x4 = sfb2d(
                        out[..., odd_idx_h, odd_idx_w],
                        lh[..., odd_idx_h, odd_idx_w],
                        hl[..., odd_idx_h, odd_idx_w],
                        hh[..., odd_idx_h, odd_idx_w],
                        filts, j, mode=self.mode
                    )

                    x2 = lowlevel.roll(x2, 1, dim=3)
                    x3 = lowlevel.roll(x3, 1, dim=2)
                    x4 = lowlevel.roll(x4, 1, dim=2)
                    x4 = lowlevel.roll(x4, 1, dim=3)

                    out[..., indices_h, indices_w] = (x1 + x2 + x3 + x4) / 4

        return out