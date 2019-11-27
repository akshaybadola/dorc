import torch


class Tag:
    def __init__(self, x):
        self.tag = x
        self._members = []

    @property
    def members(self):
        return self._members

    def __call__(self, f):
        # if self.tag not in f.__dict__:
        #     f.__dict__[self.tag] = True
        #     self._funcs.append(f)
        # if callable(f):
        #     if f not in self._members:
        #         self._members.append(f)
        #     return f
        # else:
        #     self._members.append(f)
        if f not in self._members:
            self._members.append(f)
        return f


control = Tag("control")
prop = Tag("prop")


class ProxyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self._indices = indices
        self._dataset = dataset

    def __getitem__(self, indx):
        return self._dataset[self._indices[indx]]

    def __len__(self):
        return len(self._indices)
