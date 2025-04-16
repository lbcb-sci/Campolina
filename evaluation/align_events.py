import os
import argparse

import polars as pl

import numpy as np
from tqdm import tqdm

from bam_utils import BamIndex
from kmer_model_utils import *

import concurrent.futures

from multiprocessing import Pool, Array

from Bio import Seq

pl.enable_string_cache(enable=True)
pl.Config.set_fmt_str_lengths(38)
os.environ['POLARS_MAX_THREADS'] = '32'

import time


def expand_borders(borders):
    return [list(range(borders[i], borders[i + 1])) for i in
                               range(len(borders) - 1)]


def create_matrix(remora_borders, prediction_borders):
    corner_x, corner_y = 0, 0
    remora_pred_matrix = np.zeros((len(remora_borders), len(prediction_borders)))
    for i in range(len(remora_borders)):
        for j in range(len(prediction_borders)):
            min_len = min(
                len(remora_borders[i]), len(prediction_borders[j]))
            if min_len == 0:
                remora_pred_matrix[i, j] = 0
                continue
            remora_pred_matrix[i, j] = len(
                set(remora_borders[i]).intersection(prediction_borders[j])) / min_len
            if remora_pred_matrix[i, j] > 0:
                corner_x, corner_y = i, j
    return remora_pred_matrix, corner_x, corner_y


def intersection_length(list1, list2):
    return len(set(list1).intersection(list2))

def solve_one_remora_event(args):
    #vector_intersection = np.vectorize(intersection_length, otypes=[float], signature='(n),(m)->(1)')
    #return vector_intersection(remora_event, predicted_events)
    remora_event, predicted_events = args
    #matrix[rowidx] = [intersection_length(remora_event, pred_event) for pred_event in predicted_events]
    return [len(set(remora_event).intersection(pred_event)) for pred_event in predicted_events]


def fill_intersection_row(args):
    idx, num_rows, remora_event, predicted_events = args
    #intersection_counts = np.frombuffer(shared_matrix_base.get_obj()).reshape((num_rows, len(predicted_events)))
    res = [len(set(remora_event).intersection(pred_event)) for pred_event in predicted_events]
    intersection_counts[idx*len(predicted_events):(idx+1)(len(predicted_events))] = res


def alternative_create_matrix(remora_borders, prediction_borders):
    remora_borders = np.array(remora_borders, dtype=object)
    prediction_borders = np.array(prediction_borders, dtype=object)

    #intersection_counts = np.zeros((len(remora_borders), len(prediction_borders)))
    vector_intersection = np.vectorize(intersection_length, otypes=[float])
    start = time.time()
    intersection_counts = vector_intersection(np.array(remora_borders, dtype=object)[:, None], np.array(prediction_borders, dtype=object))
    end = time.time()
    print(f'Matrix creation took: {end - start}')
    #print(f'Matrices are the same: {np.array_equal(intersection_counts2, intersection_counts)}')

    remora_lens, prediction_lens = np.vectorize(len)(remora_borders), np.vectorize(len)(prediction_borders)
    cartesian_lens = tuple(np.meshgrid(prediction_lens, remora_lens))
    remora_lens_ravel, prediction_lens_ravel = cartesian_lens[1].ravel(), cartesian_lens[0].ravel()
    #cartesian_lens = np.column_stack([arr.ravel() for arr in cartesian_lens])
    min_lengths = np.minimum(remora_lens_ravel, prediction_lens_ravel)
    min_lengths = np.reshape(min_lengths, (len(remora_borders), len(prediction_borders)))

    remora_pred_matrix = np.divide(intersection_counts, min_lengths, out=np.zeros_like(intersection_counts),
                                   where=min_lengths != 0)

    nonzero_indices = np.argwhere(remora_pred_matrix > 0)
    if len(nonzero_indices) > 0:
        corner_x, corner_y = nonzero_indices[-1]

    return remora_pred_matrix, corner_x, corner_y


def init(intersection_counts_a):
    global intersection_counts
    intersection_counts = intersection_counts_a

def parallelized_create_matrix(remora_borders, prediction_borders):
    #intersection_counts_base = np.zeros((len(remora_borders) * len(prediction_borders)))
    #intersection_counts = Array('d', intersection_counts_base)
    # Create a numpy array backed by shared memory
    # intersection_counts = np.frombuffer(intersection_counts_base.get_obj()).reshape((len(remora_borders), len(prediction_borders)))
    start = time.time()
    args = [(e, prediction_borders) for i, e in enumerate(remora_borders)]
    with Pool(processes=64) as pool:
        results = pool.map(solve_one_remora_event, args)
    #intersection_counts = np.reshape(intersection_counts, (len(remora_borders), len(prediction_borders)))
    #for (i, l) in results:
    #    intersection_counts[i] = l
    #intersection_counts = np.concatenate(results, axis=0)
    end = time.time()
    print(f'Parallelized extraction took: {end - start}')
    start = time.time()
    intersection_counts = np.array(results, dtype=float)
    end = time.time()
    print(f'Matrix np.array creation took: {end - start}')

    start = time.time()
    remora_borders = np.array(remora_borders, dtype=object)
    prediction_borders = np.array(prediction_borders, dtype=object)
    end = time.time()
    print(f'Conversion of borders to arrays took: {end - start}')
    start = time.time()
    remora_lens, prediction_lens = np.vectorize(len)(remora_borders), np.vectorize(len)(prediction_borders)
    cartesian_lens = tuple(np.meshgrid(prediction_lens, remora_lens))
    remora_lens_ravel, prediction_lens_ravel = cartesian_lens[1].ravel(), cartesian_lens[0].ravel()
    #cartesian_lens = np.column_stack([arr.ravel() for arr in cartesian_lens])
    end = time.time()
    print(f'Preparation for min length took: {end - start}')
    start = time.time()
    min_lengths = np.minimum(remora_lens_ravel, prediction_lens_ravel)
    min_lengths = np.reshape(min_lengths, (len(remora_borders), len(prediction_borders)))
    end = time.time()
    print(f'Actually getting min length took: {end - start}')

    start = time.time()
    np.divide(intersection_counts, min_lengths, out=intersection_counts,
                                   where=min_lengths != 0)
    end = time.time()
    print(f'Division took: {end - start}')

    nonzero_indices = np.argwhere(intersection_counts > 0)
    if len(nonzero_indices) > 0:
        corner_x, corner_y = nonzero_indices[-1]

    return intersection_counts, corner_x, corner_y

def traceback(align_matrix, corner_x, corner_y):
    i, j = corner_x, corner_y
    aligned_pairs = [(j, i)]
    match_ref = {}
    match_query = {}
    insertion_d = {}
    deletion_d = {}

    while i > 0 and j > 0:
        diagonal, insertion, deletion = align_matrix[i-1, j-1], align_matrix[i, j-1], align_matrix[i-1, j]
        maxval = max(diagonal, insertion, deletion)
        if maxval == diagonal:
            if i in insertion_d:
                insertion_d[i].append(j)
            elif j in deletion_d:
                deletion_d[j].append(i)
            else:
                match_query[j] = i
                match_ref[i] = j
            i -= 1
            j -= 1
        elif maxval == insertion:
            if i not in insertion_d:
                insertion_d[i] = []
            insertion_d[i].append(j)
            j -= 1
        else:
            if j not in deletion_d:
                deletion_d[j] = []
            deletion_d[j].append(i)
            i -= 1
        aligned_pairs.append((j, i))
    aligned_pairs = sorted(aligned_pairs, key=lambda x: (x[0], x[1]))
    return aligned_pairs, match_query, match_ref, insertion_d, deletion_d


def convert_reverse_mapping(aligned_pairs, ref_seq, query_len, ref_len):
    """

    :param aligned_pairs: pairs of relative (query, sequence) positions in alignment; if there is an insertion or a deletion None is set as corresponding value
    :param ref_seq: forward reference sequence
    :return:
    """
    ref_seq = str(Seq.Seq(ref_seq).reverse_complement())
    converted_pairs = []
    for i, j in aligned_pairs:
        ii = query_len - i - 1 if i is not None else None
        jj = ref_len - j - 1 if j is not None else None
        converted_pairs.append((jj, ii))
    converted_pairs = converted_pairs[::-1]
    return converted_pairs, ref_seq


def find_kmer_levels(aligned_seq, aligned_ref, kmer_model):
    int_query = seq_to_int(aligned_seq.upper())
    int_ref = seq_to_int(aligned_ref.upper())

    query_levels = kmer_model.extract_levels(int_query)
    ref_levels = kmer_model.extract_levels(int_ref)

    return query_levels, ref_levels


def remora_kmer_extraction(remora_refinement, kmer_model):
    ref_seq_alignment = remora_refinement.get_aligned_pairs()
    relative_ref_seq_alignment = [(j - remora_refinement.reference_start, i) if j is not None else (None, i) for i, j in ref_seq_alignment]
    ref_seq = remora_refinement.get_reference_sequence()
    if not remora_refinement.is_forward:
        relative_ref_seq_alignment, ref_seq = convert_reverse_mapping(relative_ref_seq_alignment, ref_seq, remora_refinement.infer_query_length(),remora_refinement.reference_length)
    aligned_seq = remora_refinement.get_forward_sequence()

    query_levels, ref_levels = find_kmer_levels(aligned_seq, ref_seq, kmer_model)

    return query_levels, ref_levels, relative_ref_seq_alignment, aligned_seq, ref_seq


def solve_pair_alignment(args):
    our_idx, remora_idx, ref_levels, kmer_model, ref_seq, prediction_borders, tmp_kmer_inverse, event_means, match_ids, deletion_ids = args
    if remora_idx < kmer_model.center_idx or remora_idx > len(ref_seq) - kmer_model.bases_after:
        return None
    event_start, event_end = prediction_borders[our_idx], prediction_borders[our_idx + 1]
    kmer_level = ref_levels[remora_idx]
    basecalled_kmer = ref_seq[
                      remora_idx - kmer_model.bases_before: remora_idx + kmer_model.kmer_len - kmer_model.bases_before].upper()
    event_mean = event_means[our_idx]
    event_align_status = 0 if our_idx in match_ids else (1 if our_idx in deletion_ids else 2)
    levels = np.array(list(tmp_kmer_inverse.keys()))
    level_diffs = levels - event_mean
    nearest_kmer_level = levels[np.argmin(np.abs(level_diffs))]
    nearest_kmer = tmp_kmer_inverse[nearest_kmer_level]
    return (event_start, event_end, event_mean, kmer_level, basecalled_kmer, nearest_kmer, event_align_status)

def get_event_kmer_alignment(event_means, prediction_borders, remora_borders, prediction_remora_alignment, match_ids, deletion_ids, query_levels, ref_levels, relative_ref_seq_alignment, aligned_seq, ref_seq, kmer_model):
    """
    :param event_means: Normalized event means
    :param prediction_borders: Event borders obtained from our predictions
    :param remora_borders: Event borders obtained through Remora refinement
    :param prediction_remora_alignment: Alignment of our events and Remora events
    :param query_levels: Normalized kmer levels for basecalled sequence
    :param ref_levels: Normalized kmer levels for reference sequence
    :param relative_ref_seq_alignment: Aligned pairs of basecalled sequence and reference sequence (indices of events)
    :param aligned_seq: Actual basecalled sequence
    :param ref_seq: Actual reference sequence
    :param kmer_model:
    :return:
    """
    #event_details = []  # this serves as a structure in which we store kmer details for each of our events (event_start, event_end, mean value of signal event, normalized kmer level from reference, basecalled kmer, match/insertion/deletion event alignment)
    tmp_kmer_inverse = dict((v, k) for k, v in kmer_model.str_kmer_levels.items())
    args = [(our_idx, remora_idx, ref_levels, kmer_model, ref_seq, prediction_borders, tmp_kmer_inverse, event_means,
             match_ids, deletion_ids) for our_idx, remora_idx in prediction_remora_alignment]
    with Pool(64) as pool:
        results = pool.map(solve_pair_alignment, args)
    event_details = [r for r in results if r is not None]
    #event_details = sorted(event_details, key=lambda x: x[0])
    schema = {'event_start': pl.Int32, 'event_end': pl.Int32, 'event_mean': pl.Float32, 'ref_kmer_level': pl.Float32,
              'ref_kmer': pl.Categorical, 'nearest_table_kmer': pl.Categorical, 'event_align_status': pl.Categorical}
    event_details = pl.LazyFrame(event_details, schema=schema)
    return event_details


def main(args):
    kmer_model = kmerModel(kmer_model_filename = args.kmer_model)
    refined_bam = BamIndex(args.remora_bam)

    predictions = pl.scan_csv(args.predictions)
    read_ids = sorted(predictions.select(pl.col('read_id')).unique().collect()['read_id'].to_list() if args.read_ids is None else args.read_ids)

    schema = {'event_start': pl.Int32, 'event_end': pl.Int32, 'event_mean': pl.Float32, 'ref_kmer_level': pl.Float32,
              'ref_kmer': pl.Categorical, 'nearest_table_kmer': pl.Categorical, 'event_align_status': pl.Categorical,
              'read_id': pl.Categorical}
    full_kmer_info = pl.LazyFrame(schema=schema)

    tqdm.write('Starting read processing')
    for read_id in tqdm(read_ids):
        print(read_id)
        for a in refined_bam.get_alignment(read_id):
            start = time.time()
            query_levels, ref_levels, relative_ref_seq_alignment, aligned_seq, ref_seq = remora_kmer_extraction(a, kmer_model)
            end = time.time()
            print(f'Remora kmer extraction took: {end - start}')
            remora_borders = np.array(a.get_tag('RR')) + a.get_tag('ts')

        prediction_borders = np.array(predictions.filter(pl.col('read_id') == read_id).select('event_start').collect()['event_start'].to_list())
        event_means = np.array(predictions.filter(pl.col('read_id') == read_id).select('event_mean').collect()['event_mean'].to_list())
        event_means = (event_means - np.mean(event_means)) / np.std(event_means)

        start = time.time()
        expanded_remora_borders = expand_borders(remora_borders)
        end = time.time()
        print(f'Expanding Remora borders took: {end - start}')
        start = time.time()
        expanded_prediction_borders = expand_borders(prediction_borders)
        end = time.time()
        print(f'Expanding our borders took: {end - start}')

        """start = time.time()
        align_matrix, corner_x, corner_y = create_matrix(expanded_remora_borders, expanded_prediction_borders)
        end = time.time()
        print(f'Creating matrix took: {end - start}')"""
        start = time.time()
        align_matrix2, corner_x2, corner_y2 = parallelized_create_matrix(expanded_remora_borders, expanded_prediction_borders)
        end = time.time()
        print(f'Alternative creating matrix took: {end - start}')

        #print((align_matrix == align_matrix2).all())

        start = time.time()
        aligned_pairs, match_query, match_ref, insertion, deletion = traceback(align_matrix2, corner_x2, corner_y2)
        end = time.time()
        print(f'Traceback took: {end - start}')

        start = time.time()
        event_kmer_alignment = get_event_kmer_alignment(event_means, prediction_borders, remora_borders, aligned_pairs, list(match_query.keys()), list(deletion.keys()), query_levels, ref_levels, relative_ref_seq_alignment, aligned_seq, ref_seq, kmer_model)
        end = time.time()
        print(f'Event kmer alignment took: {end - start}')
        event_kmer_alignment = event_kmer_alignment.with_columns(read_id = pl.lit(read_id).cast(pl.Categorical))
        full_kmer_info = pl.concat([full_kmer_info, event_kmer_alignment])

    full_kmer_info.collect().write_csv(args.tgt_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--remora_bam', default='/mnt/sod2-project/csb4/wgs/sara/event_detection_root/zymo_r104/sample_BS_r10_dorado_refined.bam')
    parser.add_argument('--predictions', default='/mnt/sod2-project/csb4/wgs/sara/event_detection_root/zymo_r104/small_test_raw_prominent_pipeline/noniter_events.csv')
    parser.add_argument('--read_ids', default=['04a49b6e-a8f9-458a-b132-763f5628cbbf'], nargs='+')
    #parser.add_argument('--read_ids', default=None, nargs='+')
    parser.add_argument('--kmer_model', default='/home/bakics/remora_analysis/9mer_levels_v1.txt')
    parser.add_argument('--tgt_file', default='/home/bakics/scratch/segmentation_pipeline_test/noniter_kmer_info.csv')

    main(parser.parse_args())