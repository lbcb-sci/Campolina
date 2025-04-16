from signal import signal

import numpy as np
import pandas as pd
import polars as pl
import torch


def convert_to_full_signal_system(chunk_peaks, chunk_starts, read_ids, chunks, mode='raw'):
    signal_peaks = [peaks + start for peaks, start in zip(chunk_peaks, chunk_starts)]
    signals = []
    full_signal_peaks = []
    if mode == 'analysis':
        signal_change = np.where(np.array(read_ids)[:-1] != np.array(read_ids[1:]))[0] + 1
        if len(signal_change) == 0:
            signals = [np.concatenate(chunks)]
            full_signal_peaks = [np.concatenate(signal_peaks)]
        else:
            signal_change = np.concatenate((np.array([0]), signal_change, np.array([len((chunks))])))
            signals = [np.concatenate(chunks[signal_change[i]:signal_change[i+1]]) for i in range(0, len(signal_change)-1)]
            full_signal_peaks = [np.concatenate(signal_peaks[signal_change[i]:signal_change[i+1]]) for i in range(0, len(signal_change)-1)]
    return signal_peaks, full_signal_peaks, signals


def convert_to_full_signal_system2(chunk_peaks, chunk_starts, read_ids, chunks, mode='raw'):
    # Compute signal peaks directly
    signal_peaks = [peaks + start for peaks, start in zip(chunk_peaks, chunk_starts)]
    #signal_peaks = [signal_p[np.insert(np.diff(signal_p) > 1, 0, True)] for signal_p in signal_peaks]
    return signal_peaks, [], []

    """# Convert lists to arrays for efficient slicing
    read_ids = np.array(read_ids)
    chunks = np.array(chunks, dtype=object)
    signal_peaks = np.array(signal_peaks, dtype=object)

    # Detect transitions in read_ids
    signal_change = np.where(read_ids[:-1] != read_ids[1:])[0] + 1
    signal_change = np.concatenate(([0], signal_change, [len(chunks)]))

    # Use slicing for grouping signals and peaks
    signals = [np.concatenate(chunks[signal_change[i]:signal_change[i + 1]]) for i in range(len(signal_change) - 1)]
    full_signal_peaks = [np.concatenate(signal_peaks[signal_change[i]:signal_change[i + 1]]) for i in range(len(signal_change) - 1)]

    return signal_peaks, full_signal_peaks, signals"""


def process_raw_output_format2(peaks, chunk_borders, read_ids, chunks):
    # Convert to full signal system
    signal_peaks, _, _ = convert_to_full_signal_system2(peaks, chunk_borders, read_ids, chunks, mode='raw')

    # Vectorize read_id expansion
    read_ids = np.array(read_ids)
    #signal_peaks = [torch.tensor(np.array(p), dtype=torch.int32) for p in signal_peaks]
    full_rids = np.repeat(read_ids, [len(peaks) for peaks in signal_peaks])

    # Concatenate all signal peaks into a single tensor
    full_peaks = torch.cat(signal_peaks).cpu()
    #full_peaks = full_peaks[np.insert(np.diff(full_peaks) > 1, 0, True)]

    # Create DataFrame directly
    return pd.DataFrame({'read_id': full_rids, 'event_start': full_peaks.numpy()})



def process_raw_output_format(peaks, chunk_borders, read_ids, chunks):
    signal_peaks, _, _ = convert_to_full_signal_system(peaks, chunk_borders, read_ids, chunks, mode='raw')
    full_rids = np.concatenate([[chunk_rid] * (len(chunk_peaks)) for chunk_rid, chunk_peaks in zip(read_ids, signal_peaks)])
    #full_peaks = np.concatenate(signal_peaks)
    full_peaks = torch.cat(signal_peaks).cpu()
    return pd.DataFrame({'read_id': full_rids, 'event_start': full_peaks})



def process_analysis_output_format(peaks, chunk_borders, read_ids, chunks):
    cols = {'read_id': pl.Categorical, 'event_start': pl.Int32, 'event_len': pl.Int32, 'event_mean': pl.Float32,
            'event_std': pl.Float32}
    signal_peaks, full_signal_peaks, signals = convert_to_full_signal_system(peaks, chunk_borders, read_ids, chunks, mode='analysis')
    full_rids = np.concatenate([[chunk_rid] * (len(chunk_peaks)) for chunk_rid, chunk_peaks in zip(read_ids, signal_peaks)])
    #full_peaks = np.concatenate([np.insert(p, 0, 0, axis=0) for p in signal_peaks])
    full_peaks = np.concatenate(signal_peaks)
    #full_peaks = np.concatenate((np.array([0]), full_peaks))
    try:
        signal_events = []
        for signal, peaks in zip(signals, full_signal_peaks):
            signal_events.extend(np.split(signal, peaks)[1:])
        #signal_events = np.concatenate(signal_events)
        #signal_events = np.concatenate([np.split(signal, peaks) for signal, peaks in zip(signals, signal_peaks)])
    except TypeError:
        print(f'TypeError for peaks: {signal_peaks}')
        return pl.LazyFrame(schema=cols)
    event_descriptors = [(rid, peak, len(e), np.mean(e), np.std(e)) for rid, peak, e in zip(full_rids, full_peaks, signal_events)]
    frame = pl.LazyFrame(event_descriptors, schema=cols, orient='row')
    return frame


def process_output_format(peaks, chunk_borders, read_ids, mode, signal_chunks=None):
    if mode == 'raw':
        return process_raw_output_format2(peaks, chunk_borders, read_ids, signal_chunks)
    else:
        return process_analysis_output_format(peaks, chunk_borders, read_ids, signal_chunks)
