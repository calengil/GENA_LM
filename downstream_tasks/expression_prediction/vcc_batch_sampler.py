"""Context-pure rectangular gene-chunk sampler for the VCC loss."""
from __future__ import annotations

import math
import numpy as np
from torch.utils.data import ConcatDataset


def vcc_children(dataset):
    children = dataset.datasets if isinstance(dataset, ConcatDataset) else [dataset]
    return children if children and all(getattr(x, "is_vcc_context_dataset", False) for x in children) else None


class VCCContextBatchSampler:
    def __init__(self, dataset, batch_size: int, shuffle: bool, seed: int, drop_last: bool):
        self.dataset = dataset
        self.children = vcc_children(dataset)
        if self.children is None:
            raise TypeError("VCCContextBatchSampler requires only VCC context datasets")
        self.batch_size, self.shuffle, self.seed, self.drop_last = int(batch_size), shuffle, int(seed), drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        for child in self.children:
            if hasattr(child, "set_epoch"):
                child.set_epoch(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches, offset = [], 0
        for child in self.children:
            indices = np.arange(len(child), dtype=np.int64)
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                part = indices[start:start + self.batch_size]
                if len(part) == self.batch_size or not self.drop_last:
                    batches.append((part + offset).tolist())
            offset += len(child)
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        fn = math.floor if self.drop_last else math.ceil
        return sum(fn(len(child) / self.batch_size) for child in self.children)
