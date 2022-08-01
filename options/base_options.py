"""
This code is based from
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""

import argparse
import os
# from util import util
import torch
import datetime
import json

import models
import data


class BaseOptions():
    """This class defines options used during both training and test time.
    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self, dataroot='../../data/denoising'):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        self.dataroot = dataroot
        self.checkpoints_dir = os.path.join(self.dataroot, 'checkpoints')

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--seed', type=int, default=1,
            help='random seed')
        parser.add_argument('--prefix', type=str, default='',
            help='customized suffix: opt.savedir = prefix + opt.savedir')
        parser.add_argument('--suffix', type=str, default='',
            help='customized suffix: opt.savedir = opt.savedir + suffix')

        parser.add_argument('--multi_gpu',dest='multi_gpu', action='store_true',
            help='use all GPUs in machines')
        parser.add_argument('--no_multi_gpu', dest='multi_gpu', action='store_false',
            help='do not enable multiple GPUs')
        parser.set_defaults(multi_gpu=True)
        parser.add_argument('--gpu_ids', type=int, nargs='+', default=[],
            help='gpu ids: e.g. 0  0,1,2, 0,2. use [] for CPU')
        parser.add_argument('--device', type=str, default='cpu',
            help='CPU or GPU')
        parser.add_argument('--n_threads', type=int, default=0,
            help='number of threads for data loader to use, Default: 0')

        # directories to save the results
        parser.add_argument('--data_dir', default=self.dataroot,
            help='path to images')
        parser.add_argument('--checkpoints_dir', type=str, default=self.checkpoints_dir,
            help='checkpoint directory')
        parser.add_argument('--savedir', type=str, default=None,
            help='models are saved here')

        parser.add_argument('--is_train', type=bool, default=True,
            help='phase')
        parser.add_argument('--load_epoch', type=str, default='latest',
            help='determine which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--resume', action='store_true',
            help='continue training: load the latest model')
        parser.add_argument('--url', default=False, action='store_true',
            help='download checkpoint from url')

        # model
        parser.add_argument('--model', type=str, default='waveletdl2',
            choices=[
                'redcnn', 'wavresnet', # LDCT denoising
                'wganvgg', # LDCT + perceptuality denoising
                'sacnn', 'sacnnae', 'cpce3d', 'cpce2d', # LDCT 3D denoising
                'dncnn', 'ircnn', 'ffdnet', 'cbdnet', 'ridnet', 'brdnet',
                'dhdn', 'dswn', # denoising only model
                'mlp', 'red', 'mwcnn', 'rdn', 'memnet', #  image resotration
                'rcan', # Super-resolution
                'unet', # Segmentation
                'waveletdl',
                'waveletganp', 'waveletganp3',
                'waveletwganp', 'waveletwganp2', 'waveletwganp3', # GAN + perceptual + wavelet
                'ganp', # Combine with edsr and unet
                'edsr', 'edsrwganp',  # EDSR related
                'waveletunet', 'waveletunetganp', # To compare algorithm with unet
                'selfdgan' # unsupervised learning
            ],
            help='specify a model')

        # dataset parameters
        parser.add_argument('--datasets',  nargs='+', default=['sidd2'],
            help='datasets for training')
        parser.add_argument('--test_datasets',  nargs='+', default=[],
            help='datasets for test')
        parser.add_argument('--batch_size', type=int, default=32,
            help='input batch size')
        parser.add_argument('--n_channels', type=int, default=3,
            help='# of image channels')
        parser.add_argument('--rgb_range', type=float, default=255,
            help='maximum value of RGB or pixel value')
        parser.add_argument('--patch_size', type=int, default=80,
            help='size of patch')
        parser.add_argument('--ext', type=str, default='sep',
            help='dataset file extension (sep, img, reset)')
        parser.add_argument('--in_mem', action='store_true',
            help='load whole data into memory, default: False')

        parser.add_argument('--add_noise', default=False, action='store_true',
            help='add noise to clean image to generate noisy images')
        parser.add_argument('--noise', type=int, default=0,
            help='Gaussian Noise standard deviation, if set to 0, add random noise to make blind noise')

        parser.add_argument("--test_patches", dest='test_patches', action='store_true',
            help='divide image into patches when test')
        parser.add_argument('--test_image', dest='test_patches', action='store_false',
            help='test whole image when test')
        

        parser.add_argument('--patch_offset', type=int, default=15,
            help='size of patch offset')
        # parser.set_defaults(test_patches=False)

        # additional parameters
        parser.add_argument('--verbose', action='store_true',
             help='if specified, print more debugging information')

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # Set the basic options
        base_opt, _ = parser.parse_known_args()
        if self.is_train and base_opt.resume:
            savedir = self.select_checkpoint_dir(base_opt)
            base_opt = self.load_options(base_opt, savedir)
        elif not self.is_train and not base_opt.url:
            savedir = self.select_checkpoint_dir(base_opt)
            base_opt = self.load_options(base_opt, savedir)

        # print('base_opt:', base_opt)
        # modify model-related parser options
        model_name = base_opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.is_train)
        opt, _ = parser.parse_known_args()  # parse again with new defaults
        # print('model opt:', opt)

        # modify dataset-related parser options
        for dataset in base_opt.datasets:
            dataset_name = dataset
            dataset_option_setter = data.get_option_setter(dataset_name)
            parser = dataset_option_setter(parser, self.is_train)
        
        for dataset in base_opt.test_datasets:
            dataset_name = dataset
            dataset_option_setter = data.get_option_setter(dataset_name)
            parser = dataset_option_setter(parser, self.is_train)
            
        # save and return the parser
        self.parser = parser
        opt = parser.parse_args()
        # print('opt:', opt)
        
        if not self.is_train or opt.resume:
            opt = self.load_options(opt, savedir)
            opt.savedir = savedir

        return opt

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.is_train = self.is_train   # train or test

        # # Get savedir from model or select checkpoint dir
        if self.is_train and not opt.resume:
            # Set checkpoint directory to save
            set_savedir = models.get_savedir_setter(opt.model)
            opt.savedir = set_savedir(opt)
            # print(opt.savedir)
            self.save_options(opt)
            opt.log_file = os.path.join(opt.savedir, opt.log_file)
        elif not self.is_train and opt.url:
            opt.savedir = os.path.join(opt.checkpoints_dir, opt.model)
            # print('set savedir later')
        # else:
            # We need to set option again with new parser
            # print('loading opt:', opt, end='\n\n')
            # print('opt.savedir:', opt.savedir)
            # opt = self.load_options(opt, savedir)

        # print("savedir:", os.path.abspath(opt.savedir))
        if self.is_train:
            opt.expr_dir = os.path.join(opt.savedir, opt.exprdir)
            os.makedirs(opt.expr_dir, exist_ok=True)
        else:
            opt.test_results_dir = os.path.join(opt.data_dir, 'test-results', os.path.basename(opt.savedir))
            os.makedirs(opt.test_results_dir, exist_ok=True)
        
        self.set_gpus(opt)
        self.print_options(opt)

        # set gpu ids

        self.opt = opt
        return self.opt
    
    def print_options(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: {}]'.format(str(default))
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        # expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        
        if self.is_train:
            file_name = os.path.join(opt.savedir, 'train_opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')

    """
    My methods
    """
    def save_options(self, opt):
        os.makedirs(opt.savedir, exist_ok=True)
        config_file = os.path.join(opt.savedir, "config.txt")
        with open(config_file, 'w') as f:
            json.dump(opt.__dict__, f, indent=2)

    def load_options(self, opt, savedir):
        # print(opt.savedir)
        config_file = os.path.join(savedir, "config.txt")
        with open(config_file, 'r') as f:
            # opt.__dict__ = json.load(f)
            saved_options = json.load(f)

        """
        Set parameters to be controlled
        """
        resume = opt.resume
        test_patches = opt.test_patches
        test_datasets = opt.test_datasets
        patch_offset = opt.patch_offset
        # savedir = opt.savedir
        load_epoch = opt.load_epoch
        
        for key in saved_options:
            if key in opt:
                # print("saved_options[{}]: {}".format(key, saved_options[key]))
                # print("opt[{}]: {}".format(key, opt.__dict__[key]))
                opt.__dict__[key] = saved_options[key]

        # print('in opt:', opt)

        opt.resume = resume
        opt.test_patches = test_patches
        opt.test_datasets = test_datasets
        opt.patch_offset = patch_offset
        opt.savedir = savedir
        opt.load_epoch = load_epoch
        return opt

    def select_checkpoint_dir(self, opt):
        print("checkpoint_dir:", os.path.abspath(opt.checkpoints_dir))
        dirs = sorted(os.listdir(opt.checkpoints_dir))

        for i, d in enumerate(dirs, 0):
            print("({}) {}".format(i, d))
        d_idx = input("Select directory that you want to load: ")

        path_opt = dirs[int(d_idx)]
        savedir = os.path.abspath(os.path.join(self.checkpoints_dir, path_opt))
        print("savedir: {}".format(savedir))
        return savedir

    def set_gpus(self, opt):
        n_gpu = torch.cuda.device_count()
        if opt.multi_gpu and len(opt.gpu_ids) == 0 and torch.cuda.is_available():
            opt.gpu_ids = list(range(torch.cuda.device_count()))
        elif len(opt.gpu_ids) > 0 and torch.cuda.is_available():
            gpu_ids = opt.gpu_ids
            opt.gpu_ids = []
            for id in gpu_ids:
                if id >= 0 and id < n_gpu:
                    opt.gpu_ids.append(id)
            opt.gpu_ids = sorted(opt.gpu_ids)
        else:
            opt.gpu_ids = []
        
        if len(opt.gpu_ids) > 0:
            print("Enabling GPUs", opt.gpu_ids)
            if len(opt.gpu_ids) > 1:
                opt.multi_gpu = True
            else:
                opt.multi_gpu = False
            opt.device = "cuda:{}".format(opt.gpu_ids[0])
        else:
            print("No GPUs use")
            opt.device = "cpu"