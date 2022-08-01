import os, glob
from importlib import import_module
# from torch.utils.data import dataloader
from torch.utils import data as D
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset

from data.patchdata import PatchData
from data.patchdata3d import PatchData3D

def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and (issubclass(cls, PatchData) or issubclass(cls, PatchData3D)):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


# def create_dataset(opt):
#     """Create a dataset given the option.

#     This function wraps the class CustomDatasetDataLoader.
#         This is the main interface between this package and 'train.py'/'test.py'

#     Example:
#         >>> from data import create_dataset
#         >>> dataset = create_dataset(opt)
#     """
#     data_loader = CustomDatasetDataLoader(opt)
#     dataset = data_loader.load_data()
#     return dataset

def create_dataset(opt):
    dataloader = {}
    if opt.is_train:
        # dataloader['train'] = TrainDataloader(opt).train_dataloader
        dataloader['train'], dataloader['test'] = TrainDataloader(opt).get_datasets()
    else:
        dataloader['test'] = TestDataloader(opt).test_dataloaders

    return dataloader


# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.mode = datasets[0].mode

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)

class TrainDataloader:
    def __init__(self, args):
        self.loader_train = None
        train_datasets = []
        for dataset_name in args.datasets:
            dataset_class = find_dataset_using_name(dataset_name)
            train_datasets.append(dataset_class(args, name=dataset_name))

        datasets = MyConcatDataset(train_datasets)
        valid_len = int(args.valid_ratio * len(datasets))
        train_len = len(datasets) - valid_len
        train_d, valid_d = D.random_split(datasets, lengths=[train_len, valid_len])
        print('Number of samples in datasets:', len(datasets))
        print('Number of samples in training sets:', len(train_d))
        print('Number of samples in validation sets:', len(valid_d))

        # for i in range(len(train_datasets)):
        #     print("Number of train datasets: {}".format(len(train_datasets[i])))
        self.train_dataloader = DataLoader(
            # MyConcatDataset(train_datasets)
            train_d,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=args.n_threads,
        )

        self.valid_dataloader = DataLoader(
            valid_d,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.n_threads,
        )
    
    def get_datasets(self):
        return self.train_dataloader, self.valid_dataloader


class TestDataloader:
    def __init__(self, args):

        if len(args.test_datasets) == 0:
            args.test_datasets = args.datasets

        batch_size = args.batch_size if args.test_random_patch else 1
        self.test_dataloaders = []
        # print('args.test_datasets:', args.test_datasets)
        for dataset_name in args.test_datasets:
            dataset_class = find_dataset_using_name(dataset_name)
            testset = dataset_class(args, name=dataset_name, is_train=False)
            if len(testset) != 0:
                # print("is_train:", args.is_train)
                if args.test_ratio < 1.0 and args.is_train:
                    testset_len = int(args.test_ratio * len(testset))
                    remain_len = len(testset) - testset_len
                    testset, _ = D.random_split(testset, lengths=[testset_len, remain_len])
                    print("Number of test datasets[{}]: {}".format(dataset_name, len(testset)))
                self.test_dataloaders.append(
                    DataLoader(
                        testset,
                        batch_size=batch_size,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=args.n_threads
                    )
                )