import os
import glob
import pickle
import numpy as np

# from skimage import io
# from skimage.external import tifffile
import imageio

from . import common
from .base_dataset import BaseDataset

class PatchData(BaseDataset):
    def __init__(self, args, name='', is_train=True):
        self.args = args
        self.dataset = name
        self.in_mem = args.in_mem
        # self.swt = args.swt
        # self.swt_lv = args.swt_lv
        # self.wavelet_func = args.wavelet_func
        self.n_channels = args.n_channels

        self.is_train = is_train
        # self.benchmark = benchmark
        self.test_random_patch = args.test_random_patch
        self.mode = 'train' if is_train else 'test'

        self.add_noise = args.add_noise
        self.noise = 0
        
        self._set_filesystem(args.data_dir)
        print("----------------- {} {} dataset -------------------".format(name, self.mode))
        print("Set file system for {} dataset {}".format(self.mode, self.dataset))
        print("apath:", os.path.abspath(self.apath))
        print("dir_hr:", os.path.abspath(self.dir_hr))
        print("dir_lr:", os.path.abspath(self.dir_lr))
        print("----------------- End ---------------------------")

        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)

        list_hr, list_lr = self._scan()

        if args.ext.find('img') >= 0:
            print("{} image loading".format(__file__))
            self.images_hr, self.images_lr = list_hr, list_lr
        elif args.ext.find('sep') >= 0:
            self.images_hr, self.images_lr = [], []
            for h in list_hr:
                b = h.replace(self.apath, path_bin)
                os.makedirs(os.path.dirname(b), exist_ok=True)

                b = b.replace(self.ext[0], '.pt')
                self.images_hr.append(b)
                self._check_and_load(args.ext, h, b, verbose=True) 
            for l in list_lr:
                b = l.replace(self.apath, path_bin)
                os.makedirs(os.path.dirname(b), exist_ok=True)

                b = b.replace(self.ext[1], '.pt')
                self.images_lr.append(b)
                self._check_and_load(args.ext, l, b, verbose=True)

            if self.in_mem:
                self._load2mem()
            
        if self.is_train:
            n_patches = args.batch_size * args.test_every
            n_images = len(args.datasets) * len(self.images_hr)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)

    # Below functions as used to prepare images
    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        names_lr = sorted(
            glob.glob(os.path.join(self.dir_lr, '*' + self.ext[1]))
        )

        return names_hr, names_lr

    def _set_filesystem(self, data_dir):
        self.apath = os.path.join(data_dir, self.mode, self.dataset)
        self.dir_hr = os.path.join(self.apath, 'hr')
        self.dir_lr = os.path.join(self.apath, 'lr')
        
    def _check_and_load(self, ext, img, f, verbose=True):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose:
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                # if self.args.n_channels == 1:
                #     pickle.dump(t_imread(img), _f)
                # else : 
                #     pickle.dump(imread(img), _f)
                pickle.dump(imageio.imread(img), _f)

    def __getitem__(self, idx):
        if not self.in_mem:
            lr, hr, filename = self._load_file(idx)
        else:
            lr, hr, filename = self._load_mem(idx)

        if self.is_train or self.test_random_patch:
            pair = self.get_patch(lr, hr)
        else:
            pair = [lr, hr]

        pair = common.set_channel(*pair, n_channels=self.n_channels)
        # if self.n_channels == 3: pair = [(p.astype(np.float) / 255.0) for p in pair]
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)

        return pair_t[0], pair_t[1], filename
            
    def __len__(self):
        if self.is_train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.is_train:
            return idx % len(self.images_hr)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        if self.args.ext == 'img':
            # if self.args.n_channels == 1 :
            #     hr = t_imread(f_hr)
            #     lr = t_imread(f_lr)
            # else : 
            #     hr = imread(f_hr)
            #     lr = imread(f_lr)
            hr = imageio.imread(f_hr)
            lr = imageio.imread(f_lr)

        elif self.args.ext.find('sep') >= 0:
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f)
            with open(f_lr, 'rb') as _f:
                lr = pickle.load(_f)
                
        hr = np.asarray(hr)
        lr = np.asarray(lr)
            
        return lr, hr, filename

    def _load_mem(self, idx):
        idx = self._get_index(idx)
        lr = self.images_lr[idx]
        hr = self.images_hr[idx]
        filename = self.filename_list[idx]

        return lr, hr, filename

    def _load2mem(self):
        images_hr_list = []
        images_lr_list = []
        self.filename_list = []
        for f_hr, f_lr in zip(self.images_hr, self.images_lr):
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f)
            with open(f_lr, 'rb') as _f:
                lr = pickle.load(_f)
            images_hr_list.append(hr)
            images_lr_list.append(lr)
            filename, _ = os.path.splitext(os.path.basename(f_hr))
            self.filename_list.append(filename)

        self.images_hr = images_hr_list
        self.images_lr = images_lr_list

    def get_patch(self, lr, hr):
        lr, hr = common.get_patch(
            lr, hr,
            patch_size=self.args.patch_size,
            n_channels=self.n_channels
        )
        if self.args.augment: lr, hr = common.augment(lr, hr)
        if self.add_noise: lr = common.add_noise(lr, self.noise)

        return lr, hr
