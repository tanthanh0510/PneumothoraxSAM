from .detection import BaseDetectionDataset
from .instance_seg import BaseInstanceDataset
from .semantic_seg import (
    BaseSemanticDataset,
    VOCSemanticDataset,
    TorchVOCSegmentation,
    PneumothoraxDataset,
    PneumoSampler
)
from .transforms import get_transforms

segment_datasets = {'base_ins': BaseInstanceDataset, 'base_sem': BaseSemanticDataset,
                    'voc_sem': VOCSemanticDataset, 'torch_voc_sem': TorchVOCSegmentation,
                    'PneumothoraxDataset': PneumothoraxDataset}

samplers = {'PneumoSampler': PneumoSampler}

det_dataset = {'base_det': BaseDetectionDataset, }


def get_dataset(cfg):
    name = cfg.name
    assert name in segment_datasets or name in det_dataset, \
        print('{name} is not supported, please implement it first.'.format(name=name))
    transform = get_transforms(cfg.transforms)
    if name in det_dataset:
        return det_dataset[name](**cfg.params, transform=transform)
    target_transform = get_transforms(cfg.target_transforms)
    return segment_datasets[name](**cfg.params, transform=transform, target_transform=target_transform)


def get_sampler(cfg):
    if cfg:
        name = cfg.name
        assert name in samplers, \
            print(
                '{name} is not supported, please implement it first.'.format(name=name))
        return samplers[name](**cfg.params)


class Iterator:
    def __init__(self, loader):
        self.loader = loader
        self.init()

    def init(self):
        self.iterator = iter(self.loader)

    def get(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            self.init()
            data = next(self.iterator)

        return data
