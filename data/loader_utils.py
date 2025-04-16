from pod5 import DatasetReader
from pathlib import Path
from torch.utils.data import IterableDataset, DataLoader

import multiprocessing as mp


class SignalDataset(IterableDataset):
    def __init__(self, queue: mp.Queue):
        super().__init__()

        self.queue = queue

    def __iter__(self):
        while (read := self.queue.get()) is not None:
            yield read


def reader_worker(queue: mp.Queue, pod5_dir: Path, workers: int):
    with DatasetReader(pod5_dir, True) as reader:
        for read in reader:
            queue.put((str(read.read_id), read.signal))

    for _ in range(workers):
        queue.put(None)
