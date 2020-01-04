import torch
import numpy as np


def default_collate_fn(batch):
    """Collates the data according to index (unzips)"""
    output = [[] for _ in batch[0]]
    for x in batch:
        for i in range(len(x)):
            output[i].append(x[i])
    return output


def default_tensorify(batch):
    """Converts each item of batch to tensor and stacks them"""
    batch = default_collate_fn(batch)
    return [torch.stack(torch.Tensor(x)) for x in batch]


class MyDataLoader:
    def __init__(self, dataset, batch_size, return_raw=True, shuffle=False, *args, **kwargs):
        self.dataset = dataset
        self.return_raw = return_raw
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.dataset))
        if "collate_fn" in kwargs:
            self.collate_fn = kwargs["collate_fn"]
        else:
            self.collate_fn = default_collate_fn
        if "drop_last" in kwargs:
            self.drop_last = kwargs["drop_last"]
        else:
            self.drop_last = False
        if self.shuffle:
            np.random.shuffle(self.indices)

    def get_next_indices(self):
        if not len(self.indices):
            return
        elif self.batch_size > len(self.indices):
            inds = self.indices.copy()
            self.indices = []
        else:
            inds = np.random.choice(self.indices, self.batch_size, False)
            self.indices = np.setdiff1d(self.indices, inds)
        return inds

    def __len__(self):
        if self.drop_last and len(self.indices) % self.batch_size:
            return len(self.indices) // self.batch_size
        else:
            return (len(self.indices) // self.batch_size) + 1

    # batch is sent as a list. Maybe send as tensor? Not sure
    def __next__(self):
        batch_indices = self.get_next_indices()
        if batch_indices:
            batch = self.collate_fn([self.dataset[i] for i in batch_indices])
            if self.return_raw:
                raw = [self.dataset._get_raw(i) for i in batch_indices]
                return raw, batch
            else:
                return batch
        else:
            return

    def __iter__(self):
        return self
