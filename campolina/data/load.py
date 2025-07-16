import logging
import numpy as np

from .pod5_util import process_chunk, get_reads
from .bam_index import BamIndex

def load_batches_seq(
        bam_idx: BamIndex, 
        pod5_path: str, 
        batch_size: int, 
    ): 
    """
    Load data for training or validation.
    """

    logger = logging.getLogger('load_batches')

    current_batch = []; current_borders = []

    for read in get_reads(pod5_path):

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

            yield np.array(current_batch), np.array(current_borders)

            remaining = len(signal_chunks) - to_take

            while remaining >= batch_size:
                current_batch = signal_chunks[to_take:to_take+batch_size]
                current_borders = chunk_borders[to_take:to_take+batch_size]

                yield np.array(current_batch), np.array(current_borders)

                to_take = to_take + batch_size
                remaining = remaining - batch_size

            current_batch = signal_chunks[to_take:]
            current_borders = chunk_borders[to_take:]

        else:
            current_batch.extend(signal_chunks)
            current_borders.extend(chunk_borders)

    yield np.array(current_batch), np.array(current_borders)