import pod5 as p5
import numpy as np
import re

from scipy import stats
from Bio import Seq
import time

import torch

from numpy.lib.stride_tricks import sliding_window_view


def calibrate(signal, scale, offset):
    return np.array(scale * (signal + offset), dtype=np.float)


def get_reads(p5_path, read_ids=None):
    with p5.Reader(p5_path) as p5_reader:
        yield from p5_reader.reads(selection=read_ids, preload='samples')

def get_reads_from_pod5(p5, read_ids=None):
    yield from p5.reads(selection=read_ids, preload='samples')


def adjust_borders(borders, ref_seq, signal_len, adjust_type):
    problematic_kmers = ['AAA', 'AAG', 'ATT', 'AGA', 'AGG', 'TAC', 'TTT', 'CAA', 'CAG', 'CTT', 'CCC', 'CGA', 'CGG', 'GAG', 'GTT', 'GGA', 'GGG']
    problematic_kmers_regex = '|'.join(problematic_kmers)
    border_shift = 5
    #ref_3mers = [ref_seq[i:i+3] for i in range(border_shift, len(ref_seq) - 3)]
    problematic_kmers_border_indices = [m.start() + 2 for m in re.finditer(f'(?=({problematic_kmers_regex}))', ref_seq) if m.start() >= border_shift]
    binary_borders = np.zeros(signal_len)
    if adjust_type == 'remove':
        borders = np.delete(borders, problematic_kmers_border_indices)
        binary_borders[borders] = 1
        return binary_borders
    elif adjust_type == 'expand':
        binary_borders[borders] = 1
        binary_borders[problematic_kmers_border_indices] = 2
        return binary_borders
    else:
        #print('Adjustment options are: "remove" and "expand". Returning non-adjusted borders')
        binary_borders[borders] = 1
    return binary_borders


def diff1(sig):
    diffs = np.diff(sig, prepend=0)
    return diffs

def diff1_gpu(sig):
    """
    Compute the first-order difference of a signal with zero prepended.

    Parameters:
        sig (torch.Tensor): Input signal of shape (batch_size, sequence_length).

    Returns:
        torch.Tensor: First-order differences with zero prepended.
    """
    zero_prepended = torch.zeros((sig.shape[0], 1), device=sig.device, dtype=sig.dtype)
    padded_sig = torch.cat((zero_prepended, sig), dim=1)
    diffs = padded_sig[:, 1:] - padded_sig[:, :-1]
    return diffs


def window_mean_std(sig, wlen):
    sig = sliding_window_view(sig, window_shape=wlen, axis=1)
    w_means = np.mean(sig, axis=2)
    w_stds = np.std(sig, axis=2)

    zero_array = np.expand_dims(np.array([0] * sig.shape[0]), axis=1)
    w_means = np.concatenate((zero_array, np.concatenate((w_means, zero_array), axis=1)), axis=1)
    w_stds = np.concatenate((zero_array, np.concatenate((w_stds, zero_array), axis=1)), axis=1)

    return w_means, w_stds


def window_mean_std_gpu(sig, wlen):
    windows = sig.unfold(dimension=1, size=wlen, step=1)
    w_means = windows.mean(dim=-1)
    w_stds = windows.std(dim=-1)

    # Add zero padding
    zero_array = torch.zeros(sig.size(0), 1, device=sig.device, dtype=sig.dtype)
    w_means = torch.cat((zero_array, w_means, zero_array), dim=1)
    w_stds = torch.cat((zero_array, w_stds, zero_array), dim=1)

    return w_means, w_stds


def comp_cumsum_gpu(sig):
    """
    Compute cumulative sums and squared cumulative sums for a batch of signals.

    Parameters:
        sig (torch.Tensor): Input signal of shape (batch_size, sequence_length).

    Returns:
        tuple: Cumulative sums and squared cumulative sums, both as torch.Tensors.
    """
    batch_size = sig.shape[0]

    # Add zero padding to the start of the signal
    zero_array = torch.zeros((batch_size, 1), device=sig.device, dtype=sig.dtype)
    padded_sig = torch.cat((zero_array, sig), dim=1)

    # Compute cumulative sums and squared cumulative sums
    cumsum_sig = torch.cumsum(padded_sig, dim=1)
    cumsum_sig_square = torch.cumsum(padded_sig ** 2, dim=1)

    return cumsum_sig, cumsum_sig_square

def comp_tstat_gpu(cumsum_sig, cumsum_sig_square, s_len, w_len):
    """
    Compute the t-statistics for a batch of signals using cumulative sums.

    Parameters:
        cumsum_sig (torch.Tensor): Cumulative sums of the signal.
        cumsum_sig_square (torch.Tensor): Cumulative squared sums of the signal.
        s_len (int): Total length of the sequence.
        w_len (int): Window length.

    Returns:
        torch.Tensor: T-statistics of shape (batch_size, sequence_length - 1).
    """
    eta = torch.finfo(cumsum_sig.dtype).eps
    tstat = torch.zeros((cumsum_sig.shape[0], cumsum_sig.shape[1] - 1), device=cumsum_sig.device, dtype=cumsum_sig.dtype)

    # Ensure conditions are met
    if s_len < 2 * w_len or w_len < 2:
        return tstat

    # Compute cumulative sums for each window
    sum1 = cumsum_sig[:, w_len:s_len - w_len + 1] - cumsum_sig[:, :s_len - 2 * w_len + 1]
    sumsq1 = cumsum_sig_square[:, w_len:s_len - w_len + 1] - cumsum_sig_square[:, :s_len - 2 * w_len + 1]

    sum2 = cumsum_sig[:, 2 * w_len:s_len + 1] - cumsum_sig[:, w_len:s_len - w_len + 1]
    sumsq2 = cumsum_sig_square[:, 2 * w_len:s_len + 1] - cumsum_sig_square[:, w_len:s_len - w_len + 1]

    # Means for each segment
    mean1 = sum1 / w_len
    mean2 = sum2 / w_len

    # Calculate variances, ensuring a minimum threshold eta
    combined_var = (sumsq1 / w_len - mean1 ** 2) + (sumsq2 / w_len - mean2 ** 2)
    combined_var = torch.maximum(combined_var, torch.tensor(eta, device=cumsum_sig.device, dtype=cumsum_sig.dtype))

    # Compute t-statistics
    delta_mean = mean2 - mean1
    tstat_res = torch.abs(delta_mean) / torch.sqrt(combined_var / w_len)

    # Place the computed t-statistics into the result array
    tstat[:, w_len:s_len - w_len + 1] = tstat_res

    return tstat



def comp_cumsum(sig):
    batch_size = sig.shape[0]

    zero_array = np.expand_dims(np.array([0] * batch_size), axis=1)
    cumsum_sig = np.cumsum(np.concatenate((zero_array, sig), axis=1), axis=1)
    cumsum_sig_square = np.cumsum(np.concatenate((zero_array, sig ** 2), axis=1), axis=1)

    return cumsum_sig, cumsum_sig_square


def comp_tstat(cumsum_sig, cumsum_sig_square, s_len, w_len):
    eta = np.finfo(float).eps
    tstat = np.zeros((cumsum_sig.shape[0], cumsum_sig.shape[1] - 1))

    # Ensure conditions are met
    if s_len < 2 * w_len or w_len < 2:
        return tstat

    # Compute cumulative sums for each window in a vectorized manner
    sum1 = cumsum_sig[:, w_len:s_len - w_len + 1] - cumsum_sig[:, :s_len - 2 * w_len + 1]
    sumsq1 = cumsum_sig_square[:, w_len:s_len - w_len + 1] - cumsum_sig_square[:, :s_len - 2 * w_len + 1]

    sum2 = cumsum_sig[:, 2 * w_len:s_len + 1] - cumsum_sig[:, w_len:s_len - w_len + 1]
    sumsq2 = cumsum_sig_square[:, 2 * w_len:s_len + 1] - cumsum_sig_square[:, w_len:s_len - w_len + 1]

    # Means for each segment
    mean1 = sum1 / w_len
    mean2 = sum2 / w_len

    # Calculate variances, handling minimum threshold eta
    combined_var = (sumsq1 / w_len - mean1 ** 2) + (sumsq2 / w_len - mean2 ** 2)
    combined_var = np.maximum(combined_var, eta)

    # Compute t-statistics
    delta_mean = mean2 - mean1
    tstat_res = np.abs(delta_mean) / np.sqrt(combined_var / w_len)

    # Place the computed t-statistics into the result array
    tstat[:, w_len:s_len - w_len + 1] = tstat_res

    return tstat


def comp_tstat_old(cumsum_sig, cumsum_sig_square, s_len, w_len):
    eta = np.finfo(float).eps
    tstat = np.zeros((cumsum_sig.shape[0], cumsum_sig.shape[1]-1))

    if s_len < 2 * w_len or w_len < 2:
        return tstat

    for i in range(w_len, s_len - w_len + 1):
        sum1 = np.copy(cumsum_sig[:, i])
        sumsq1 = np.copy(cumsum_sig_square[:, i])
        if i > w_len:
            sum1 -= cumsum_sig[:, i - w_len]
            sumsq1 -= cumsum_sig_square[:, i - w_len]

        cumsumnext = cumsum_sig[:, i + w_len]
        cumsum_square_next = cumsum_sig_square[:, i + w_len]
        cumsum_current = cumsum_sig[:, i]
        cumsum_square_current = cumsum_sig_square[:, i]
        sum2 = cumsumnext - cumsum_current
        sumsq2 = cumsum_square_next - cumsum_square_current

        mean1 = sum1 / w_len
        mean2 = sum2 / w_len
        combined_var = sumsq1 / w_len - mean1 ** 2 + sumsq2 / w_len - mean2 ** 2
        combined_var = np.maximum(combined_var, eta)

        delta_mean = mean2 - mean1
        tstat_res = np.abs(delta_mean) / np.sqrt(combined_var / w_len)
        tstat[:, i] = tstat_res

    return tstat



def process_chunk(aln, read, adjust_type=None, predict=False, chunk_len=6000, w_len=3):
    start1 = time.time()
    signal = read.signal
    borders = np.array(aln.get_tag('RR'), dtype=np.uint32) + aln.get_tag('ts')
    borders = borders[np.where(borders < len(signal))]
    corrected_start = borders[0]
    corrected_end = borders[-1]

    ref_seq = aln.get_reference_sequence() if aln.is_forward else str(Seq.Seq(aln.get_reference_sequence()).reverse_complement())
    binary_borders = adjust_borders(borders, ref_seq, len(signal), adjust_type)

    signal = signal[corrected_start:corrected_end]
    binary_borders = binary_borders[corrected_start:corrected_end]

    signal_chunks = np.split(signal, range(chunk_len, len(signal), chunk_len))
    signal_chunks = [stats.zscore(c) for c in signal_chunks]
    chunk_borders = np.split(binary_borders, range(chunk_len, len(binary_borders), chunk_len))
    for i in range(len(chunk_borders)):
        if np.isnan(signal_chunks[i]).any():
            chunk_borders = chunk_borders[:i]
            signal_chunks = signal_chunks[:i]
            break
        
    if len(chunk_borders[-1]) < chunk_len:
        padding = np.zeros(chunk_len - len(chunk_borders[-1]))
        chunk_borders[-1] = np.concatenate((chunk_borders[-1], padding))
        signal_chunks[-1] = np.concatenate((signal_chunks[-1], padding))

    if predict:
        chunk_indices = range(corrected_start, corrected_end, chunk_len)
        identifiers = [f'{str(read.read_id)}_{chunk_indices[i]}_{chunk_indices[i+1]}' for i in range(len(chunk_indices)-1)]
        identifiers.append(f'{str(read.read_id)}_{chunk_indices[-1]}_{corrected_end}')
    else:
        identifiers = None

    signal_chunks = np.array(signal_chunks)
    cumsum_sig, cumsum_sig_square = comp_cumsum(signal_chunks)
    tstat1 = comp_tstat(cumsum_sig, cumsum_sig_square, chunk_len, w_len)
    diff = diff1(signal_chunks)
    w_means, w_stds = window_mean_std(signal_chunks, wlen=3)
    signal_chunks = list(np.stack((signal_chunks, diff, w_means, w_stds, tstat1), axis=1))

    return signal_chunks, chunk_borders, identifiers



def process_chunk2(aln, read, adjust_borders=None, predict=False, chunk_len=6000):
    signal = read.signal
    start = aln.get_tag('ts')
    borders = np.array(aln.get_tag('RR')) + start

    if adjust_borders is not None:
        binary_borders = adjust_borders(borders, aln.get_reference_sequence(), len(signal), adjust_borders)
    else:
        binary_borders = np.zeros(len(signal))
        binary_borders[borders] = 1

    corrected_start = borders[0]
    corrected_end = borders[-1]
    borders = borders[np.where(borders < corrected_end)[0]]

    #if str(read.read_id) == '1f169841-8c8b-4f57-b20c-04a03d42f106':
    #    print(start, corrected_start, corrected_end)

    signal = signal[corrected_start:corrected_end]
    #signal_chunks = [signal[i : i + chunk_len] for i in range(0, len(signal), chunk_len)]
    signal_chunks = np.split(signal, range(chunk_len, len(signal), chunk_len))
    signal_chunks = [stats.zscore(c) for c in signal_chunks]
    
    #if str(read.read_id) == '1f169841-8c8b-4f57-b20c-04a03d42f106':
    #    print(signal_chunks)

    borders = borders - corrected_start
    #if str(read.read_id) == '1f169841-8c8b-4f57-b20c-04a03d42f106':
    #    print(borders)
    if len(borders) == 0 or len(signal) == 0:
        return None, None, None
    binary_borders = np.zeros(len(signal))
    binary_borders[borders] = 1
    #chunk_borders = [binary_borders[i:i+chunk_len] for i in range(0, len(binary_borders), chunk_len)]
    chunk_borders = np.split(binary_borders, range(chunk_len, len(binary_borders), chunk_len))
    for i in range(len(chunk_borders)):
        if np.isnan(signal_chunks[i]).any():
            chunk_borders = chunk_borders[:i]
            signal_chunks = signal_chunks[:i]
            break
    if len(chunk_borders[-1]) < chunk_len:
        padding = np.zeros(chunk_len - len(chunk_borders[-1]))
        chunk_borders[-1] = np.concatenate((chunk_borders[-1], padding))
        signal_chunks[-1] = np.concatenate((signal_chunks[-1], padding))
        #if str(read.read_id) == '1f169841-8c8b-4f57-b20c-04a03d42f106':
        #    print(signal_chunks)

    if predict:
        chunk_indices = range(corrected_start, corrected_end, chunk_len)
        identifiers = [f'{str(read.read_id)}_{chunk_indices[i]}_{chunk_indices[i+1]}' for i in range(len(chunk_indices)-1)]
        identifiers.append(f'{str(read.read_id)}_{chunk_indices[-1]}_{corrected_end}')
    else:
        identifiers = None

    return signal_chunks, chunk_borders, identifiers


def get_pod5_readid_pairs(path, recursive=False):
    if path.is_file():
        return [path]

    # Finding all input POD5 files
    if recursive:
        files = path.glob(f'**/*.pod5')
    else:
        files = path.glob(f'*.pod5')

    pod5_file_readid_pairs = []
    for f in files:
        with p5.Reader(f) as reader:
            pod5_file_readid_pairs.append((f, reader.read_ids))

    return pod5_file_readid_pairs

