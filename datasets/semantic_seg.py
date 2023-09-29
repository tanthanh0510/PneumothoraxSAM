import os

import numpy as np
import pandas as pd
import random

from PIL import Image
from skimage import io
import torch

from torch.utils.data import Dataset
from torchvision.datasets import VOCSegmentation, VisionDataset


class PneumothoraxDataset(VisionDataset):
    """
    ├── input
        │   ├── my_dataset
        │   │   ├── mask
        │   │   │   ├── xxx{img_suffix}
        │   │   │   ├── yyy{img_suffix}
        │   │   │   ├── zzz{img_suffix}
        │   │   ├── train
        │   │   │   ├── xxx{ann_suffix}
        │   │   │   ├── yyy{ann_suffix}
        │   │   │   ├── zzz{ann_suffix}
        |   |   ├── train.csv
        |   |   ├── val.csv
    """

    def __init__(self, dataset_dir, usePromt,
                 image_set='train',
                 data_prefix: dict = dict(img_path='train', ann_path='mask'),
                 return_dict=False):
        super(PneumothoraxDataset, self).__init__(root=dataset_dir)
        self.class_names = ['pneumothorax']
        self.usePromt = usePromt
        self.dataset = pd.read_csv(
            os.path.join(dataset_dir, image_set + '.csv'))
        # self.dataset = self.dataset[self.dataset['existLabel'] == 1]
        self.dataset_dir = dataset_dir
        self.return_dict = return_dict
        self.img_folder_name = os.path.join(
            dataset_dir, data_prefix['img_path'])
        self.ann_folder_name = os.path.join(
            dataset_dir, data_prefix['ann_path'])
        print(
            f'img_folder_name: {self.img_folder_name}, ann_folder_name: {self.ann_folder_name}')

    def __getitem__(self, index):
        name = self.dataset.iloc[index, 0]
        img_name = os.path.join(self.img_folder_name, name)
        img = io.imread(img_name)
        if len(img.shape) == 2:
            img = np.repeat(img[:, :, None], 3, axis=-1)
        else:
            img = img
        img = np.transpose(img, (2, 0, 1))
        img = img / 255.0
        assert (
            np.max(img) <= 1.0 and np.min(img) >= 0.0
        ), "image should be normalized to [0, 1]"
        ann_file_name = os.path.join(self.ann_folder_name, name)
        ann = io.imread(ann_file_name)
        ann = ann/255
        H, W = ann.shape
        if self.usePromt != 0:
            bboxes = np.array([0, 0, W, H])
        else:
            if ann.max() == 1:
                y_indices, x_indices = np.where(ann > 0)
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                x_min = max(0, x_min - random.randint(0, 20))
                x_max = min(W, x_max + random.randint(0, 20))
                y_min = max(0, y_min - random.randint(0, 20))
                y_max = min(H, y_max + random.randint(0, 20))
                bboxes = np.array([x_min, y_min, x_max, y_max])
            else:
                bboxes = np.array([0, 0, W, H])
        print("bboxes", bboxes)
        return (
            torch.tensor(img).float(),
            torch.tensor(ann[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            name
        )

    def __len__(self):
        return len(self.dataset)


class PneumoSampler(torch.utils.data.Sampler):
    def __init__(self, dataset_dir,
                 image_set='train',
                 demand_non_empty_prob=0.8):
        assert demand_non_empty_prob > 0, 'frequency of non-empty images must be greater than zero'
        self.positive_prob = demand_non_empty_prob
        print("dataset_dir", dataset_dir)
        self.dataset = pd.read_csv(
            os.path.join(dataset_dir, image_set + '.csv'))

        self.positive_indices = self.dataset[self.dataset['existLabel']
                                             == 1].index.values
        self.negative_indices = self.dataset[self.dataset['existLabel']
                                             == 0].index.values

        self.n_positive = self.positive_indices.shape[0]
        self.n_negative = int(
            self.n_positive * (1 - self.positive_prob) / self.positive_prob)
        print('n_positive: {n_positive}, n_negative: {n_negative}'.format(
            n_positive=self.n_positive, n_negative=self.n_negative))

    def __iter__(self):
        negative_sample = np.random.choice(
            self.negative_indices, self.n_negative)
        shuffled = np.random.permutation(
            np.hstack((negative_sample, self.positive_indices)))
        return iter(shuffled.tolist())

    def __len__(self):
        return self.n_positive + self.n_negative
