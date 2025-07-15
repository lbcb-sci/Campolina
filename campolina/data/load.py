import numpy as np
from tqdm import tqdm

from .pod5_util import process_chunk, get_reads
from .bam_index import BamIndex

def load_batches(bam_idx: BamIndex, pod5_path: str, batch_size: int, predict: bool = False): 
    """
    Load data for training or validation.
    """
    # TODO what if i preload a batch of batch_size signals and then yield batches as long as i can and then load batch_size signal again?
    current_batch = []; current_borders = []; current_identifiers = []

    for read in get_reads(pod5_path):
        for alignment in bam_idx.get_alignment(str(read.read_id)):
            if alignment is None: continue

            signal_chunks, chunk_borders, chunk_identifiers = process_chunk(
                aln=alignment, 
                read=read, 
                predict=predict, 
                adjust_type=None,
            )

            if signal_chunks is None:
                tqdm.write(f'Could not extract info for read {read.read_id}')
                continue

            #tqdm.write(f'Signal {read.read_id} has {len(signal_chunks)}')

            if len(current_batch) + len(signal_chunks) > batch_size:

                to_take = batch_size - len(current_batch)
                current_batch.extend(signal_chunks[:to_take])
                current_borders.extend(chunk_borders[:to_take])

                if predict:
                    current_identifiers.extend(chunk_identifiers[:to_take])
                    yield np.array(current_batch), np.array(current_borders), np.array(current_identifiers)
                else:
                    #print(read.read_id)
                    yield np.array(current_batch), np.array(current_borders)

                remaining = len(signal_chunks) - to_take

                while remaining >= batch_size:
                    current_batch = signal_chunks[to_take:to_take+batch_size]
                    current_borders = chunk_borders[to_take:to_take+batch_size]

                    if predict:
                        current_identifiers = chunk_identifiers[to_take:to_take+batch_size]
                        yield np.array(current_batch), np.array(current_borders), np.array(current_identifiers)
                    else:
                        #print(read.read_id)
                        yield np.array(current_batch), np.array(current_borders)

                    to_take = to_take + batch_size
                    remaining = remaining - batch_size

                current_batch = signal_chunks[to_take:]
                current_borders = chunk_borders[to_take:]

                if predict: current_identifiers = chunk_identifiers[to_take:]

            else:
                current_batch.extend(signal_chunks)
                current_borders.extend(chunk_borders)
                if predict: current_identifiers.extend(chunk_identifiers)

    if predict:
        yield np.array(current_batch), np.array(current_borders), np.array(current_identifiers)
    else:
        #print(read.read_id)
        yield np.array(current_batch), np.array(current_borders)