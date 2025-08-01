import logging
import numpy as np

import torch
import torch.multiprocessing as mp

from .bam_index import BamIndex
from .extract import get_reads, process_chunk

DONE_SIGNAL = "DONE"

def to_tensors(batch: list, borders: list) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(np.array(batch),   dtype=torch.float32, device='cpu'), 
        torch.tensor(np.array(borders), dtype=torch.float32, device='cpu'),
    )

def dataloader_process(
        bucket: list,
        pod5_path: str, 
        bam_idx: BamIndex,
        batch_size: int, 
        dataset: mp.Queue,
    ) -> None: 

    logging.basicConfig(level=logging.INFO, format='[%(processName)s] %(message)s')
    logger = logging.getLogger('load_process')
    logger.info('process started')

    current_batch = []; current_borders = []

    for i, read in enumerate(get_reads(pod5_path)):

        if not i in bucket: continue # TODO stupid but temporary

        alignment = bam_idx.get_alignment(str(read.read_id))
        if alignment is None: continue

        signal_chunks, chunk_borders, _ = process_chunk(
            aln=alignment, 
            read=read, 
            adjust_type=None,
        )

        if signal_chunks is None:
            logger.warning(f'could not extract info for read {read.read_id}')
            continue

        if len(current_batch) + len(signal_chunks) > batch_size:

            to_take = batch_size - len(current_batch)
            current_batch.extend(signal_chunks[:to_take])
            current_borders.extend(chunk_borders[:to_take])

            dataset.put(to_tensors(current_batch, current_borders))

            remaining = len(signal_chunks) - to_take

            while remaining >= batch_size:
                current_batch = signal_chunks[to_take:to_take+batch_size]
                current_borders = chunk_borders[to_take:to_take+batch_size]

                dataset.put(to_tensors(current_batch, current_borders))

                to_take = to_take + batch_size
                remaining = remaining - batch_size

            current_batch = signal_chunks[to_take:]
            current_borders = chunk_borders[to_take:]

        else:
            current_batch.extend(signal_chunks)
            current_borders.extend(chunk_borders)

    if i in bucket:
        dataset.put(to_tensors(current_batch, current_borders))

    logger.info('sending done signal')
    dataset.put((DONE_SIGNAL, None))

def load_batches_mp(
        bam_index: BamIndex,
        pod5_path: str,
        nprocesses: int = 3,
        queue_maxsize: int = 8,
        batch_size: int = 1024,
    ) -> tuple[list[mp.Process], mp.Queue]:
    '''
    Load data in parallel using multiprocessing.
    Returns the list of spawned subprocesses as well as the queue of batches that gets filled.
    '''
    manager = mp.Manager()
    dataset = manager.Queue(maxsize=queue_maxsize)

    buckets = [set() for _ in range(nprocesses)]
    for i, _ in enumerate(get_reads(pod5_path)): # TODO rewrite
        buckets[i % nprocesses].add(i)

    processes = [mp.Process(
        target=dataloader_process, 
        args=(bucket, pod5_path, bam_index, batch_size, dataset),
        name=f'DataLoader-{i+1}',
    ) for i, bucket in enumerate(buckets)]

    [process.start() for process in processes]
    return processes, dataset

