from torch.utils.data.dataloader import _utils, DataLoader, _DataLoaderIter


class MyDataLoader(DataLoader):
    def __init__(self, dataset, return_raw=False, *args, **kwargs):
        super(MyDataLoader, self).__init__(dataset, *args, **kwargs)
        self.return_raw = return_raw

    def __iter__(self):
        return _MyDataLoaderIter(self)


class _MyDataLoaderIter(_DataLoaderIter):
    def __init__(self, loader):
        self.return_raw = loader.return_raw
        super(_MyDataLoaderIter, self).__init__(loader)

    def __next__(self):
        if self.num_workers == 0:  # same-process loading
            indices = next(self.sample_iter)  # may raise StopIteration
            batch = self.collate_fn([self.dataset[i] for i in indices])
            if self.pin_memory:
                batch = _utils.pin_memory.pin_memory_batch(batch)
            if self.return_raw:
                raw = [self.dataset._get_raw(i) for i in indices]
                return raw, batch
            else:
                return batch

        # check if the next sample has already been generated
        if self.rcvd_idx in self.reorder_dict:
            batch = self.reorder_dict.pop(self.rcvd_idx)
            return self._process_next_batch(batch)

        if self.batches_outstanding == 0:
            self._shutdown_workers()
            raise StopIteration

        while True:
            assert (not self.shutdown and self.batches_outstanding > 0)
            idx, batch = self._get_batch()
            self.batches_outstanding -= 1
            if idx != self.rcvd_idx:
                # store out-of-order samples
                self.reorder_dict[idx] = batch
                continue
            return self._process_next_batch(batch)
