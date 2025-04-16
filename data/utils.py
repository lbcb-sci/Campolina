import os
import tqdm
import torch

import numpy as np
from click.core import batch
from scipy import stats

import polars as pl

from .pod5_util import diff1, window_mean_std, comp_cumsum, comp_tstat

import time

from torch.utils.data import Dataset, DataLoader


log = False

def merge_csvs(src, tgt_dir, abbrev, delete_src=True):
    full_frame = pl.concat([pl.scan_csv(writer_outf) for writer_outf in src])
    full_frame.collect().write_csv(f'{tgt_dir}/{abbrev}_events.csv')
    if delete_src:
        for path in src:
            os.remove(path)


def get_raw_batch4(loader, bs, chunk_len=6000, log=False):
    def zscore_tensor(tensor, dim=0):
        tensor = tensor.float()
        mean = torch.mean(tensor, dim=dim, keepdim=True)
        std = torch.std(tensor, dim=dim, keepdim=True)
        return (tensor - mean) / (std + 1e-8)

    def process_chunks(signal, chunk_len):
        borders = list(range(chunk_len, len(signal), chunk_len))
        nonnorm_chunks = [signal[i:j] for i, j in zip([0] + borders, borders + [len(signal)])]
        borders = [0] + borders

        if len(nonnorm_chunks) > 1:
            chunks = torch.stack([
                zscore_tensor(chunk) for chunk in nonnorm_chunks[:-1]
            ])

            final_chunk = torch.cat([
                zscore_tensor(nonnorm_chunks[-1]),
                torch.zeros(chunk_len - len(nonnorm_chunks[-1]))
            ])
            chunks = torch.cat((chunks, final_chunk.unsqueeze(0)), dim=0)
        else:
            chunks = torch.cat([
                zscore_tensor(nonnorm_chunks[0]),
                torch.zeros(chunk_len - len(nonnorm_chunks[0]))
            ]).unsqueeze(0)

        return chunks, borders, nonnorm_chunks


    current_batch = []
    current_chunk_borders = []
    current_read_ids = []
    current_signal_chunks = []

    for read_id, signal in loader:
        signal_chunks, borders, nonnorm_chunks = process_chunks(signal.squeeze(), chunk_len)

        for chunk_start in range(0, len(signal_chunks), bs):
            batch_chunks = signal_chunks[chunk_start:chunk_start + bs]
            batch_borders = borders[chunk_start:chunk_start + bs]
            batch_read_ids = [str(read_id)] * len(batch_chunks)
            batch_nonnorm = nonnorm_chunks[chunk_start:chunk_start + bs]

            current_batch.extend(batch_chunks)
            current_chunk_borders.extend(batch_borders)
            current_read_ids.extend(batch_read_ids)
            current_signal_chunks.extend(batch_nonnorm)

            if len(current_batch) >= bs:
                yield (current_batch[:bs],
                       current_chunk_borders[:bs],
                       current_read_ids[:bs],
                       current_signal_chunks[:bs])
                current_batch = current_batch[bs:]
                current_chunk_borders = current_chunk_borders[bs:]
                current_read_ids = current_read_ids[bs:]
                current_signal_chunks = current_signal_chunks[bs:]

    # Yield any remaining data
    if current_batch:
        yield current_batch, current_chunk_borders, current_read_ids, current_signal_chunks


def get_raw_batch3(reader, read_ids, bs, chunk_len=6000, log=False):
    def process_chunks(signal, chunk_len):
        borders = list(range(chunk_len, len(signal), chunk_len))
        nonnorm_chunks = np.split(signal, borders)
        borders = [0] + borders
        if len(nonnorm_chunks) > 1:
            chunks = stats.zscore(nonnorm_chunks[:-1], axis=1)
            final_chunk = np.concatenate(
                (stats.zscore(nonnorm_chunks[-1]), np.zeros(chunk_len - len(nonnorm_chunks[-1]))))
            chunks = np.vstack((chunks, final_chunk))
        else:
            chunks = np.expand_dims(
                np.concatenate((stats.zscore(nonnorm_chunks[0]), np.zeros(chunk_len - len(nonnorm_chunks[0])))), axis=0)
        return chunks, borders, nonnorm_chunks

    current_batch = []
    current_chunk_borders = []
    current_read_ids = []
    current_signal_chunks = []

    for r in reader.reads(selection=read_ids, preload='samples'):
        start_time = time.time()
        signal_chunks, borders, nonnorm_chunks = process_chunks(r.signal, chunk_len)
        end_time = time.time()

        for chunk_start in range(0, len(signal_chunks), bs):
            batch_chunks = signal_chunks[chunk_start:chunk_start + bs]
            batch_borders = borders[chunk_start:chunk_start + bs]
            batch_read_ids = [str(r.read_id)] * len(batch_chunks)
            batch_nonnorm = nonnorm_chunks[chunk_start:chunk_start + bs]

            current_batch.extend(batch_chunks)
            current_chunk_borders.extend(batch_borders)
            current_read_ids.extend(batch_read_ids)
            current_signal_chunks.extend(batch_nonnorm)

            if len(current_batch) >= bs:
                yield (current_batch[:bs],
                       current_chunk_borders[:bs],
                       current_read_ids[:bs],
                       current_signal_chunks[:bs])
                current_batch = current_batch[bs:]
                current_chunk_borders = current_chunk_borders[bs:]
                current_read_ids = current_read_ids[bs:]
                current_signal_chunks = current_signal_chunks[bs:]

    # Yield any remaining data
    if current_batch:
        yield current_batch, current_chunk_borders, current_read_ids, current_signal_chunks


def get_raw_batch2(reader, read_ids, bs, chunk_len=6000, log=False):
    def process_chunks(signal, chunk_len):
        borders = list(range(chunk_len, len(signal), chunk_len))
        nonnorm_chunks = np.split(signal, borders)
        borders = [0] + borders
        if len(nonnorm_chunks) > 1:
            chunks = stats.zscore(nonnorm_chunks[:-1], axis=1)
            final_chunk = np.concatenate(
                (stats.zscore(nonnorm_chunks[-1]), np.zeros(chunk_len - len(nonnorm_chunks[-1]))))
            chunks = np.vstack((chunks, final_chunk))
        else:
            chunks = np.expand_dims(
                np.concatenate((stats.zscore(nonnorm_chunks[0]), np.zeros(chunk_len - len(nonnorm_chunks[0])))), axis=0)
        return chunks, borders, nonnorm_chunks

    current_batch = []
    current_chunk_borders = []
    current_read_ids = []
    current_signal_chunks = []

    for r in reader.reads(selection=read_ids):
        start_time = time.time()
        chunks, borders, nonnorm_chunks = process_chunks(r.signal, chunk_len)

        # Feature computation
        cumsum_sig, cumsum_sig_square = comp_cumsum(chunks)
        tstat1 = comp_tstat(cumsum_sig, cumsum_sig_square, chunk_len, 3)
        diff = diff1(chunks)
        w_means, w_stds = window_mean_std(chunks, wlen=3)

        signal_chunks = np.stack((chunks, diff, w_means, w_stds, tstat1), axis=1)
        if log:
            print(f'Feature extraction took {time.time() - start_time:.2f} seconds')

        for chunk_start in range(0, len(signal_chunks), bs):
            batch_chunks = signal_chunks[chunk_start:chunk_start + bs]
            batch_borders = borders[chunk_start:chunk_start + bs]
            batch_read_ids = [str(r.read_id)] * len(batch_chunks)
            batch_nonnorm = nonnorm_chunks[chunk_start:chunk_start + bs]

            current_batch.extend(batch_chunks)
            current_chunk_borders.extend(batch_borders)
            current_read_ids.extend(batch_read_ids)
            current_signal_chunks.extend(batch_nonnorm)

            if len(current_batch) >= bs:
                yield (current_batch[:bs],
                       current_chunk_borders[:bs],
                       current_read_ids[:bs],
                       current_signal_chunks[:bs])
                current_batch = current_batch[bs:]
                current_chunk_borders = current_chunk_borders[bs:]
                current_read_ids = current_read_ids[bs:]
                current_signal_chunks = current_signal_chunks[bs:]

    # Yield any remaining data
    if current_batch:
        yield current_batch, current_chunk_borders, current_read_ids, current_signal_chunks


def get_raw_batch(reader, read_ids, bs, chunk_len=6000):
    current_batch = []
    current_chunk_borders = []
    current_read_ids = []
    current_signal_chunks = []
    for r in reader.reads(selection=read_ids):
        feature_start = time.time()
        if len(current_batch) == bs:
            yield current_batch, current_chunk_borders, current_read_ids, current_signal_chunks
            current_batch = []
            current_chunk_borders = []
            current_read_ids = []
            current_signal_chunks = []
        borders = list(range(chunk_len, len(r.signal), chunk_len))
        nonnorm_chunks = np.split(r.signal, borders)
        borders = [0] + borders
        if len(nonnorm_chunks) > 1:
            chunks = stats.zscore(nonnorm_chunks[:-1], axis=1)
            final_chunk = np.concatenate((stats.zscore(nonnorm_chunks[-1]), np.zeros(chunk_len - len(nonnorm_chunks[-1]))))
            chunks = np.concatenate((chunks, np.expand_dims(final_chunk, axis=0)))
        else:
            chunks = np.expand_dims(np.concatenate((stats.zscore(nonnorm_chunks[0]), np.zeros(chunk_len - len(nonnorm_chunks[-1])))), axis=0)
        cumsum_sig, cumsum_sig_square = comp_cumsum(chunks)
        tstat1 = comp_tstat(cumsum_sig, cumsum_sig_square, chunk_len, 3)
        diff = diff1(chunks)
        w_means, w_stds = window_mean_std(chunks, wlen=3)
        signal_chunks = list(np.stack((chunks, diff, w_means, w_stds, tstat1), axis=1))
        feature_end = time.time()
        if log:
            print(f'Feature extraction took {feature_end - feature_start}')

        batch_start = time.time()
        #TODO I need to provide multiple signals if the batch consists of multiple signals!!!
        if len(current_batch) + len(signal_chunks) > bs:
            to_take = bs - len(current_batch)
            current_batch.extend(signal_chunks[:to_take])
            current_chunk_borders.extend(borders[:to_take])
            current_read_ids.extend([str(r.read_id)]*to_take)
            current_signal_chunks.extend(nonnorm_chunks[:to_take])

            batch_end = time.time()
            if log:
                print(f'Loading batch 1 took {batch_end - batch_start}')
            yield current_batch, current_chunk_borders, current_read_ids, current_signal_chunks
            batch_start = time.time()

            remaining = len(signal_chunks) - to_take
            while remaining >= bs:
                current_batch = signal_chunks[to_take:to_take+bs]
                current_chunk_borders = borders[to_take:to_take+bs]
                current_read_ids = [str(r.read_id)]*bs
                current_signal_chunks = nonnorm_chunks[to_take:to_take+bs]

                batch_end = time.time()
                if log:
                    print(f'Batch loading 2 took {batch_end - batch_start}')
                yield current_batch, current_chunk_borders, current_read_ids, current_signal_chunks
                batch_start = time.time()

                to_take += bs
                remaining -= bs

            current_batch = signal_chunks[to_take:]
            current_chunk_borders = borders[to_take:]
            current_read_ids = [str(r.read_id)]*(len(signal_chunks) - to_take)
            current_signal_chunks = nonnorm_chunks[to_take:]
        else:
            current_batch.extend(signal_chunks)
            current_chunk_borders.extend(borders)
            current_read_ids.extend([str(r.read_id)]*(len(signal_chunks)))
            current_signal_chunks.extend(nonnorm_chunks)
    yield current_batch, current_chunk_borders, current_read_ids, current_signal_chunks


def raw_chunk_signal(read_ids, reader, chunk_len=6000):
    for r in reader.reads(selection=read_ids):
        #start = time.time()
        #if len(r.signal) < chunk_len:
            #tqdm.tqdm.write(f'Signal with read id {r.read_id} with length {len(r.signal)} is too short to be processed. Continuing...')
            #continue
        borders = list(range(chunk_len, len(r.signal), chunk_len))
        chunks = [stats.zscore(c) for c in np.split(r.signal, borders)]
        #chunks = np.split(r.signal, borders)
        padding = np.zeros(chunk_len - len(chunks[-1]))
        chunks[-1] = np.concatenate((chunks[-1], padding))
        chunks = np.array(chunks)
        cumsum_sig, cumsum_sig_square = comp_cumsum(chunks)
        tstat1 = comp_tstat(cumsum_sig, cumsum_sig_square, chunk_len, 3)
        diff = diff1(chunks)
        w_means, w_stds = window_mean_std(chunks, wlen=3)
        #end = time.time()
        #print(f'Raw chunk preprocessing took: {end - start}')
        signal_chunks = list(np.stack((chunks, diff, w_means, w_stds, tstat1), axis=1))
        yield signal_chunks, [0] + borders, str(r.read_id), r.signal


def concat_back_to_signal(chunk_peaks, chunk_starts, signal_len):
    signal_peaks = np.concatenate([peaks + start for peaks, start in zip(chunk_peaks, chunk_starts)])
    signal_peaks = signal_peaks[np.where((signal_peaks < signal_len) & (signal_peaks > 0))]
    #signal = np.concatenate(chunks[:-1])
    #final_peaks_corrected = chunk_peaks[-1] + chunk_starts[-1]
    #final_peaks_corrected = final_peaks_corrected[np.where(final_peaks_corrected >= chunk_starts[-2] + 6000)]
    #final_chunk = chunks[-1][chunk_starts[-2] + 6000:]
    #signal_peaks = np.concatenate((signal_peaks, final_peaks_corrected))
    #signal = np.concatenate((signal, final_chunk))
    return signal_peaks


def process_signal_output_format(signal, all_peaks, chunk_starts, read_id):
    cols = {'read_id': pl.Categorical, 'event_start': pl.Int32, 'event_len': pl.Int32, 'event_mean': pl.Float32,
            'event_std': pl.Float32}
    signal_peaks = concat_back_to_signal(all_peaks, chunk_starts, len(signal))
    try:
        signal_events = np.split(signal, signal_peaks)
    except TypeError:
        print(f'TypeError for peaks: {signal_peaks}')
        return pl.LazyFrame(schema=cols)
    #signal_peaks = np.concatenate((np.array([0]), signal_peaks))
    #signal_peaks = np.split(signal_peaks, np.where(np.diff(signal_peaks) != 1)[0] + 1) #TODO this now removed to keep consecutive predictions as well
    #print(signal_peaks)
    #signal_peaks = np.concatenate([cons[0::3] for cons in signal_peaks]) #TODO this now removed to keep consecutive predictions as well
    #print(signal_peaks)
    try:
        signal_events = np.split(signal, signal_peaks)
    except TypeError:
        print(f'TypeError for peaks: {signal_peaks}')
        return pl.LazyFrame(schema=cols)
    signal_peaks = np.concatenate((np.array([0]), signal_peaks))
    #print(len(signal_peaks), len(np.unique(signal_peaks)))
    event_descriptors = [(read_id, signal_peak, len(e), np.mean(e), np.std(e)) for signal_peak, e in zip(signal_peaks, signal_events)]      #TODO want to make this faster, myb put into dataframe and then apply mean, std, len to the column
    frame = pl.LazyFrame(event_descriptors, schema=cols, orient='row')
    #collected = frame.collect()
    #event_descriptors.schema = cols
    return frame
