import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from file_io import Events, read_2d_spikes

###############################################################################
# This file is adapted from PySNN (https://github.com/BasBuller/PySNN).
# A small modification was made to address some errors in N-MNIST data loading.
###############################################################################

#########################################################
# Utility functions
#########################################################
def _train_test_split_classification(datasets, labels=None, train_size=0.8, seed=42):
    if not isinstance(datasets, (list, tuple)):
        datasets = [datasets]
        labels = [labels]
    train = []
    test = []
    lbls = ["sample", "label"]

    # Split each data subset into desired train test
    for idx, dset in enumerate(datasets):
        if labels[idx] is not None:
            x_trn, x_tst, y_trn, y_tst = train_test_split(
                dset, labels[idx], train_size=train_size, random_state=seed
            )
            trn = pd.DataFrame({lbls[0]: x_trn, lbls[1]: y_trn})
            tst = pd.DataFrame({lbls[0]: x_tst, lbls[1]: y_tst})
        else:
            trn, tst = train_test_split(dset, train_size=train_size, random_state=seed)
            if not isinstance(trn, pd.DataFrame):
                trn = pd.DataFrame(trn, columns=lbls)
                tst = pd.DataFrame(tst, columns=lbls)
            else:
                trn.columns = lbls
                tst.columns = lbls
                trn.reset_index(inplace=True, drop=True)
                tst.reset_index(inplace=True, drop=True)

        train.append(trn)
        test.append(tst)

    train = pd.concat(train)
    test = pd.concat(test)

    return train, test


def _list_dir_content(root_dir):
    content = {}
    for root, dirs, files in os.walk(root_dir):
        for label, name in enumerate(dirs):
            subdir = os.path.join(root, name)
            dir_content = os.listdir(subdir)
            content[name] = [os.path.join(subdir, im) for im in dir_content]
    return content


def _concat_dir_content(content):
    ims = []
    labels = []
    names = []
    for idx, (name, data) in enumerate(content.items()):
        if not isinstance(data, (list, tuple)):
            data = [data]
        ims += data
        labels += [idx for _ in range(len(data))]
        names += [name for _ in range(len(data))]
    df = pd.DataFrame({"sample": ims, "label": labels})
    return df, names


def train_test(root_dir, train_size=0.8, seed=42):
    r"""Split dataset into train and test sets.
    
    Takes in a directory where it looks for sub-directories. Content of each directory is split into train and test subsets.

    :param root_dir: Directory containing data.
    :param train_size: Percentage of the data to be assigned to training set, 1 - train_size is assigned as test set.
    :param seed: Seed for random number generator.
    """
    content = _list_dir_content(root_dir)
    data, _ = _concat_dir_content(content)
    train, test = _train_test_split_classification(
        data, train_size=train_size, seed=seed
    )
    return train, test


class NeuromorphicDataset(Dataset):
    r"""Class that wraps around several neuromorphic datasets.

    The class adheres to regular PyTorch dataset conventions.

    :param data: Data for dataset formatted in a pd.DataFrame.
    :param sampling_time: Duration of interval between samples.
    :param sample_length: Total duration of a single sample.
    :param height: Number of pixels in height direction.
    :param width: Number of pixels in width direction.
    :param im_transform: Image transforms, same convention as for PyTorch datasets.
    :param lbl_transform: Lable transforms, same convention as for PyTorch datasets.
    """

    def __init__(
        self,
        data,
        sampling_time,
        sample_length,
        height,
        width,
        im_transform=None,
        lbl_transform=None,
    ):
        self.data = data
        self.im_transform = im_transform
        self.lbl_transform = lbl_transform

        self.sampling_time = sampling_time
        self.sample_length = sample_length
        self.height = height
        self.width = width
        self.n_time_bins = int(sample_length / sampling_time)
        #self.im_template = torch.zeros((2, height, width, self.n_time_bins))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Image sample
        im_name = self.data.iloc[idx, 0]
        input_spikes = read_2d_spikes(im_name)
        sample = input_spikes.to_spike_tensor(
            torch.zeros((2, self.height, self.width, self.n_time_bins)), sampling_time=self.sampling_time
        )

        # Label
        label = self.data.iloc[idx, 1]

        # Apply transforms
        if self.im_transform:
            sample = self.im_transform(sample)
        if self.lbl_transform:
            label = self.lbl_transform(label)

        return (sample, label)

#########################################################
# Neuromorphic MNIST
#########################################################
def nmnist_train_test(
    root,
    sampling_time=1,
    sample_length=300,
    height=34,
    width=34,
    im_transform=None,
    lbl_transform=None,
):
    r"""Neurmorphic version of the MNIST dataset, obtained from:

        https://www.garrickorchard.com/datasets/n-mnist
    
    'Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades' by G. Orchard et al.

    :return: :class:`NeuromorphicDataset` for both and training and test data.
    """
    train_content = _list_dir_content(os.path.join(root, "Train"))
    train, _ = _concat_dir_content(train_content)
    train_dataset = NeuromorphicDataset(
        train,
        sampling_time,
        sample_length,
        height,
        width,
        im_transform=im_transform,
        lbl_transform=lbl_transform,
    )

    test_content = _list_dir_content(os.path.join(root, "Test"))
    test, _ = _concat_dir_content(test_content)
    test_dataset = NeuromorphicDataset(
        test,
        sampling_time,
        sample_length,
        height,
        width,
        im_transform=im_transform,
        lbl_transform=lbl_transform,
    )

    return train_dataset, test_dataset