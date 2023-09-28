from .semantic_seg import (
    PneumothoraxDataset,
    PneumoSampler
)
from .transforms import get_transforms

segment_datasets = {'PneumothoraxDataset': PneumothoraxDataset}

samplers = {'PneumoSampler': PneumoSampler}


def get_dataset(cfg):
    name = cfg.name
    assert name in segment_datasets, \
        print('{name} is not supported, please implement it first.'.format(name=name))
    transform = get_transforms(cfg.transforms)
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
