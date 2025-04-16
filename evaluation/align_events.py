import multiprocessing
import os
import argparse
import re

import polars as pl

import numpy as np
from tqdm import tqdm
from collections import Counter

from bam_utils import BamIndex
from kmer_model_utils import *

import multiprocessing as mp

import pod5 as p5
from Bio import Seq

pl.enable_string_cache(enable=True)
pl.Config.set_fmt_str_lengths(38)
os.environ['POLARS_MAX_THREADS'] = '32'


def convert_reverse_mapping(aligned_pairs, ref_seq, query_len, ref_len):
    """

    :param aligned_pairs: pairs of relative (query, sequence) positions in alignment; if there is an insertion or a deletion None is set as corresponding value
    :param ref_seq: forward reference sequence
    :return:
    """
    ref_seq = str(Seq.Seq(ref_seq).reverse_complement())
    converted_pairs = []
    for i, j in aligned_pairs:
        ii = ref_len - i - 1 if i is not None else None
        jj = query_len - j - 1 if j is not None else None
        converted_pairs.append((ii, jj))
    converted_pairs = converted_pairs[::-1]
    return converted_pairs, ref_seq


def adjust_borders(borders, ref_seq, ref_kmers, ref_levels):
    problematic_kmers = ['AAA', 'AAG', 'ATT', 'AGA', 'AGG', 'TAC', 'TTT', 'CAA', 'CAG', 'CTT', 'CCC', 'CGA', 'CGG',
                         'GAG', 'GTT', 'GGA', 'GGG']
    problematic_kmers_regex = '|'.join(problematic_kmers)
    border_shift = 5
    # ref_3mers = [ref_seq[i:i+3] for i in range(border_shift, len(ref_seq) - 3)]
    problematic_kmers_border_indices = [m.start() + 2 for m in re.finditer(f'(?=({problematic_kmers_regex}))', ref_seq)
                                        if ((m.start() >= border_shift) & (m.start() < (len(ref_seq) - 3 - 1)))]
    problematic_kmers_indices = np.array(problematic_kmers_border_indices) - border_shift - 1
    problematic_levels_indices = np.array(problematic_kmers_border_indices)
    borders = np.delete(borders, problematic_kmers_border_indices)
    ref_kmers = np.delete(ref_kmers, problematic_kmers_indices)
    ref_levels = np.delete(ref_levels, problematic_levels_indices)
    return borders, ref_kmers, ref_levels

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


def expand_borders(borders):
    return [list(range(borders[i], borders[i + 1])) for i in
                               range(len(borders) - 1)]


def intersection_length(list1, list2):
    return len(set(list1).intersection(list2))


def create_matrix(remora_borders, prediction_borders):
    remora_borders = np.array(remora_borders, dtype=object)
    prediction_borders = np.array(prediction_borders, dtype=object)

    #intersection_counts = np.zeros((len(remora_borders), len(prediction_borders)))
    vector_intersection = np.vectorize(intersection_length, otypes=[float])
    #start = time.time()
    intersection_counts = vector_intersection(np.array(remora_borders, dtype=object)[:, None], np.array(prediction_borders, dtype=object))
    #end = time.time()
    #print(f'Matrix creation took: {end - start}')
    #print(f'Matrices are the same: {np.array_equal(intersection_counts2, intersection_counts)}')

    if len(remora_borders) == 0 or len(prediction_borders) == 0:
        print(remora_borders)
        print(prediction_borders)
        return None, None, None
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
    else:
        corner_x, corner_y = 0, 0

    return remora_pred_matrix, corner_x, corner_y


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



def solve_pair_alignment(args):
    read_id, our_idx, remora_idx, ref_levels, kmer_model, ref_kmers, prediction_borders, remora_borders, tmp_kmer_inverse, event_means, remora_means, match_ids, deletion_ids = args
    if remora_idx < kmer_model.center_idx or remora_idx > len(ref_kmers):
        return None
    event_start, event_end = prediction_borders[our_idx], prediction_borders[our_idx + 1]
    remora_start, remora_end = remora_borders[remora_idx], remora_borders[remora_idx+1]
    if remora_idx >= len(ref_kmers):
        print(f'Read {read_id}, with {len(ref_kmers)} kmers, trying for index {remora_idx}')
    kmer_level = ref_levels[remora_idx]
    basecalled_kmer = ref_kmers[remora_idx - kmer_model.bases_before]
    #basecalled_kmer = ref_seq[
    #                  remora_idx - kmer_model.bases_before: remora_idx + kmer_model.kmer_len - kmer_model.bases_before].upper()
    event_mean = event_means[our_idx]
    remora_mean = remora_means[remora_idx]
    event_align_status = 0 if our_idx in match_ids else (1 if our_idx in deletion_ids else 2)
    levels = np.array(list(tmp_kmer_inverse.keys()))
    level_diffs = levels - event_mean
    nearest_kmer_level = levels[np.argmin(np.abs(level_diffs))]
    nearest_kmer = tmp_kmer_inverse[nearest_kmer_level]
    return (event_start, event_end, event_mean, remora_start, remora_end, remora_mean, kmer_level, basecalled_kmer, nearest_kmer, event_align_status)


def get_event_kmer_alignment(read_id, event_means, prediction_borders, remora_means, remora_borders, prediction_remora_alignment, match_ids, deletion_ids, ref_levels, ref_seq, kmer_model):
    tmp_kmer_inverse = dict((v, k) for k, v in kmer_model.str_kmer_levels.items())
    event_details = []
    for our_idx, remora_idx in prediction_remora_alignment:
        alignment = solve_pair_alignment((read_id, our_idx, remora_idx, ref_levels, kmer_model, ref_seq, prediction_borders, remora_borders, tmp_kmer_inverse, event_means, remora_means,
             match_ids, deletion_ids))
        if alignment is not None:
            event_details.append(alignment)
    schema = {'event_start': pl.Int32, 'event_end': pl.Int32, 'event_mean': pl.Float32, 'remora_start': pl.Int32, 'remora_end': pl.Int32, 'remora_event_mean': pl.Float32, 'ref_kmer_level': pl.Float32,
              'ref_kmer': pl.Categorical, 'nearest_table_kmer': pl.Categorical, 'event_align_status': pl.Categorical}
    event_details = pl.LazyFrame(event_details, schema=schema)
    return event_details



def get_remora_means(signal, remora_borders):
    event_means = np.array([np.mean(signal[remora_borders[i]:remora_borders[i+1]]) for i in range(len(remora_borders) - 1)])
    return (event_means - np.mean(event_means)) / np.std(event_means)


def merge_csvs(src, tgt_dir, filename_description, delete_src=True):
    full_frame = pl.concat([pl.scan_csv(writer_outf) for writer_outf in src if os.path.exists(writer_outf)])
    full_frame.collect().write_csv(f'{tgt_dir}/{filename_description}.csv')
    if delete_src:
        for path in src:
            if os.path.exists(path):
                os.remove(path)


def align_worker(args):
    refined_bam, pod5_file, predictions, kmer_model, full_kmer_info, read_id, remove_hard, trimmed, full_vs_trimmed, writer_path = args
    #tqdm.write('Starting worker')
    #predictions = pl.scan_csv(predictions_path).collect()
    alns = refined_bam.get_alignment(read_id)
    pod5_reader = p5.Reader(pod5_file)
    for a in alns:
        #start = time.time()
        if read_id == 'd9fccbda-4c05-4f4e-97b7-40757f8f4a3f':
            print(read_id, a)
        if a is None:
            return
        query_levels, ref_levels, relative_ref_seq_alignment, aligned_seq, ref_seq = remora_kmer_extraction(a,
                                                                                                            kmer_model)
        #end = time.time()
        #print(f'Remora kmer extraction took: {end - start}')
        remora_borders = np.array(a.get_tag('RR')) + a.get_tag('ts')
        remora_borders = np.unique(remora_borders)
        ref_kmers = [ref_seq[i:i + kmer_model.kmer_len] for i in range(len(ref_seq) - kmer_model.kmer_len + 1)]
        #print(f'Remora kmer extraction for read {read_id} done')

        if remove_hard:
            remora_borders, ref_kmers, ref_levels = adjust_borders(remora_borders, ref_seq, ref_kmers, ref_levels)
        for r in pod5_reader.reads(selection=[a.query_name]):
            remora_means = get_remora_means(r.signal, remora_borders)

        if trimmed or full_vs_trimmed:
            remora_start, remora_end = remora_borders[0], remora_borders[-1]
            remora_borders -= remora_start
        else:
            remora_start, remora_end = 0, 1e10

    prediction_borders = np.array(
        predictions.filter((pl.col('read_id') == read_id)).select('event_start').collect()['event_start'].to_list()
    )

    #print(f'Event means for read {read_id} calculated')

    #start = time.time()
    expanded_remora_borders = expand_borders(remora_borders)
    #end = time.time()
    #print(f'Expanding Remora borders took: {end - start}')
    #start = time.time()
    expanded_prediction_borders = expand_borders(prediction_borders)
    #end = time.time()
    #print(f'Expanding our borders took: {end - start}')

    #start = time.time()
    #print(f'Creating matrix for read: {read_id}')
    align_matrix2, corner_x2, corner_y2 = create_matrix(expanded_remora_borders, expanded_prediction_borders)
    if align_matrix2 is None:
        return
    #end = time.time()
    #print(f'Alternative creating matrix took: {end - start}')

    # print((align_matrix == align_matrix2).all())
    #print(f'Created matrix for read: {read_id}')

    #start = time.time()
    aligned_pairs, match_query, match_ref, insertion, deletion = traceback(align_matrix2, corner_x2, corner_y2)
    #end = time.time()
    #print(f'Traceback took: {end - start}')

    #start = time.time()
    #print(f'Getting event kmer alignment for read: {read_id}')

    event_means = np.array(
        predictions.filter((pl.col('read_id') == read_id) & (pl.col('event_start') <= prediction_borders[aligned_pairs[-1][0]])).select('event_mean').collect()['event_mean'].to_list()
    )
    event_means = (event_means - np.mean(event_means)) / np.std(event_means)


    event_kmer_alignment = get_event_kmer_alignment(read_id, event_means, prediction_borders, remora_means, remora_borders, aligned_pairs, list(match_query.keys()),
                             list(deletion.keys()), ref_levels, ref_kmers, kmer_model)
    #end = time.time()
    #print(f'Event kmer alignment took: {end - start}')
    event_kmer_alignment = event_kmer_alignment.with_columns(read_id=pl.lit(read_id).cast(pl.Categorical))
    full_kmer_info = pl.concat([full_kmer_info, event_kmer_alignment])

    full_kmer_info.collect().write_csv(writer_path)
    #print(f'Processed read: {read_id} and stored in {writer_path}')


def main(args):
    kmer_model = kmerModel(kmer_model_filename=args.kmer_model)
    refined_bam = BamIndex(args.remora_bam)

    predictions = pl.scan_csv(args.predictions)
    read_ids = sorted(predictions.select(pl.col('read_id')).unique().collect()[
                          'read_id'].to_list() if args.read_ids is None else args.read_ids)

    schema = {'event_start': pl.Int32, 'event_end': pl.Int32, 'event_mean': pl.Float32, 'remora_start': pl.Int32,
              'remora_end': pl.Int32, 'remora_event_mean': pl.Float32, 'ref_kmer_level': pl.Float32, 'ref_kmer': pl.Categorical,
              'nearest_table_kmer': pl.Categorical, 'event_align_status': pl.Categorical, 'read_id': pl.Categorical}
    full_kmer_info = pl.LazyFrame(schema=schema)


    if args.workers == 1:
        args = (refined_bam, args.pod5_file, predictions, kmer_model, full_kmer_info, args.read_ids[0], args.remove_hard, args.trimmed, args.full_vs_trimmed, f'{args.tgt_dir}/{args.filename_description}.csv')
        align_worker(args)
        exit(0)

    writers_path = [f'{args.tgt_dir}/{args.filename_description}_{i}.csv' for i in range(len(read_ids))]
    with mp.get_context("spawn").Pool(args.workers) as pool, tqdm(total=len(read_ids)) as pbar:
        worker_args = [(refined_bam, args.pod5_file, predictions, kmer_model, full_kmer_info, r, args.remove_hard, args.trimmed, args.full_vs_trimmed, writers_path[i]) for i, r in
                enumerate(read_ids)]
        for _ in pool.imap_unordered(align_worker, worker_args):
            pbar.update(1)

    tqdm.write('Merging csv files')
    merge_csvs(writers_path, args.tgt_dir, args.filename_description, args.delete_src)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--remora_bam', default='/home/bakics/scratch/Campolina_paper/segmenteval/R10_zymo_segmenteval_subset/refined_R10_zymo_segmenteval_subset.bam')
    parser.add_argument('--predictions', default='/home/bakics/scratch/Campolina_paper/segmenteval/R10_zymo_segmenteval_subset/R10_Zymo_segmenteval_14112024_Focal_alpha0_8_gamma1_alpha5000_beta0_05_eta10_5channel_final2048_400bps_events_new.csv')
    parser.add_argument('--pod5_file', default='/home/bakics/scratch/Campolina_paper/segmenteval/R10_zymo_segmenteval_subset/R10_zymo_segmenteval_subset.pod5')
    #parser.add_argument('--read_ids', default=['af83f8de-ce0e-4ed1-9e75-605304a5f74d'], nargs='+')
    parser.add_argument('--read_ids', default=None, nargs='+')
    parser.add_argument('--kmer_model', default='/home/bakics/remora_analysis/9mer_levels_v1_400bps.txt')
    parser.add_argument('--tgt_dir', default='./')
    parser.add_argument('--filename_description', default='005665c5-7b3f-44c8-989e-d0e23af14b23')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--delete_src', action='store_true')
    parser.add_argument('--remove_hard', action='store_true')
    parser.add_argument('--trimmed', action='store_true')
    parser.add_argument('--full_vs_trimmed', action='store_true')

    main(parser.parse_args())
