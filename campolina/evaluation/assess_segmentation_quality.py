import argparse
import os
import pysam
import polars as pl
import numpy as np
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors
from campolina.data import BamIndex

def load_full_events(csv_path):
    df = pl.scan_csv(csv_path).collect()
    bord_dict = {}
    for rid, g in df.group_by('read_id'):
        borders = list(g['event_start'])
        bord_dict[str(rid)] = borders
    return bord_dict

def get_remora_borders(bam_index, read_id):

    if isinstance(read_id, tuple) and len(read_id) > 0:
        read_id = read_id[0]

    #read_id = read_id.replace('(', '').replace(')', '').replace("'", '').replace(',', '')

    #for a in bam_index.get_alignment(read_id):
    alignment = bam_index.get_alignment(read_id)

    if alignment is None: return None

    remora_borders = np.array(alignment.get_tag('RR')) + alignment.get_tag('ts')

    return remora_borders

def jaccard(tp, fp, fn):
    return tp / (tp + fp + fn)

def naive_evaluation(predicted_borders, bam_index):
    # Initialize overall counters
    overall_true_borders = overall_pred_borders = 0
    overall_TP = overall_FP = overall_FN = 0
    per_read_TP, per_read_FP, per_read_FN = [], [], []
    per_read_jaccard = []

    # Iterate through the remora borders
    for key, pred_borders in predicted_borders.items():

        true_borders = get_remora_borders(bam_index, key)

        if true_borders is None:
            continue

        # Calculate overall counts
        true_border_set = set(true_borders)
        pred_border_set = set(pred_borders)

        intersection = true_border_set & pred_border_set
        tp = len(intersection)
        fp = len(pred_border_set - intersection)
        fn = len(true_border_set - intersection)

        per_read_jaccard.append(jaccard(tp, fp, fn))

    print(f'The naive Jaccard similarity is {np.mean(per_read_jaccard):.2f}')

def find_intersection(remora_borders, predicted_borders):
    count = 0
    predicted_borders = np.array(predicted_borders)
    for i in remora_borders:
        # Check if any element in b is i-1, i, or i+1
        if np.any((predicted_borders == i - 1) | (predicted_borders == i) | (predicted_borders == i + 1)):
            count += 1
    return count

def naive_expand_evaluation(predicted_borders, bam_index):
    overall_true_borders = 0
    overall_pred_borders = 0

    per_read_jaccard = []
    for k, pred_bords in predicted_borders.items():

        true_bords = get_remora_borders(bam_index, k)

        if true_bords is None:
            continue

        true_bords = set(true_bords)

        overall_true_borders += len(true_bords)
        overall_pred_borders += len(pred_bords)
        intersect_len = find_intersection(true_bords, pred_bords)
        read_FP = (len(pred_bords) - intersect_len)
        read_FN = (len(true_bords) - intersect_len)

        per_read_jaccard.append(jaccard(intersect_len, read_FP, read_FN))

    print(f'The expanded Jaccard similarity is {np.mean(per_read_jaccard)}')

def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
    """

    if direction == 'y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction == 'x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction == 'bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")

    return chamfer_dist

def eval_chamfer(full_events, bam_index):
    chamfer_total = 0
    num_samples = 0
    for rid, camp_borders in full_events.items():
        remora_border_x = get_remora_borders(bam_index, rid)
        if remora_border_x is None:
            continue
        camp_borders = np.array(camp_borders)
        camp_borders = camp_borders[
            np.where((camp_borders >= remora_border_x[0]) & (camp_borders <= remora_border_x[-1]))]
        if len(camp_borders) == 0:
            continue
        chamfer_dist = chamfer_distance(np.expand_dims(np.array(camp_borders), 1),
                                        np.expand_dims(np.array(remora_border_x), 1), metric='l1', direction='bi')
        chamfer_total += chamfer_dist
        num_samples += 1
    try:
        print(f'Bi-directional average chamfer distance: {chamfer_total / num_samples:.2f}')
    except: pass

def std_evaluation(align_df, restrict_to_matches=False, separate=False):
    align_df = align_df.filter((pl.col('remora_event_mean').is_not_nan()) & (pl.col('event_mean').is_not_nan()))
    if restrict_to_matches:
        align_df = align_df.filter(pl.col('event_align_status') == 0)
    return ((align_df['event_mean'] - align_df['remora_event_mean'])**2).sqrt().mean()

def correlation_evaluation(align_df, restrict_to_matches=False, colid='ref_kmer_level', separate=False):
    align_df = align_df.filter((pl.col('remora_event_mean').is_not_nan()) & (pl.col('event_mean').is_not_nan()))
    if restrict_to_matches:
        align_df = align_df.filter(pl.col('event_align_status') == 0)
    return pearsonr(align_df['event_mean'], align_df[colid])

def alignment_score_evaluation(match_num, insert_num, delete_num, ref_len, match_score=1, insertion_score=0.5, deletion_score=0.5):
    return (match_score*match_num - insertion_score*insert_num - deletion_score*delete_num)/ref_len

def len_alignment_score_evaluation(match_len, insert_len, delete_len):
    return match_len - (insert_len + delete_len)

def aligned_event_evaluation(align_csv):
    match_nums = []
    insert_nums = []
    delete_nums = []
    alignment_scores = []
    alignment_len_scores = []

    #align_df = pl.scan_csv(align_csv).collect()
    align_df = align_csv#.collect()

    number_remora_events = len(align_df.select(['remora_start', 'read_id']).unique())
    align_remora_ratio = len(align_df)/number_remora_events
    insertion_df = align_df.filter(pl.col("event_align_status") == 2)
    deletion_df = align_df.filter(pl.col("event_align_status") == 1)

    print(f'Ratio between overall number of alignments and number of events is {align_remora_ratio:.2f}')
    print(f'Overall match ratio: {len(align_df.filter(pl.col("event_align_status") == 0)) / number_remora_events:.2f}')
    print(f'Overall insertion ratio: {len(insertion_df.select(["remora_start", "read_id"]).unique()) / number_remora_events:.2f}')
    print(f'Overall deletion ratio: {len(deletion_df.select(["remora_start", "read_id"]).unique()) / number_remora_events:.2f}')

    for rid, g in align_df.group_by('read_id'):
        g_len = len(g)
        ref_len = len(g['remora_start'].unique())
        matches = g.filter(pl.col('event_align_status') == 0)
        insertions = g.filter(pl.col('event_align_status') == 2)
        deletions = g.filter(pl.col('event_align_status') == 1)
        match_lens = (matches['remora_end'] - matches['remora_start']).sum()
        insertion_lens = (insertions['remora_end'] - insertions['remora_start']).sum()
        deletion_lens = (deletions['remora_end'] - deletions['remora_start']).sum()
        match_nums.append(len(matches) / g_len)
        insert_nums.append(len(insertions) / g_len)
        delete_nums.append(len(deletions) / g_len)
        alignment_scores.append(alignment_score_evaluation(len(matches), len(insertions), len(deletions), ref_len))
        alignment_len_scores.append(len_alignment_score_evaluation(match_lens, insertion_lens, deletion_lens))

    full_std_eval = std_evaluation(align_df)
    match_std_eval = std_evaluation(align_df, restrict_to_matches=True)
    remora_full_correlation_eval = correlation_evaluation(align_df, restrict_to_matches=False, colid='remora_event_mean')
    remora_match_correlation_eval = correlation_evaluation(align_df, restrict_to_matches=True, colid='remora_event_mean')
    full_correlation_eval = correlation_evaluation(align_df, restrict_to_matches=False, colid='ref_kmer_level')
    match_correlation_eval = correlation_evaluation(align_df, restrict_to_matches=True, colid='ref_kmer_level')
    #print(f'Average alignment score: {np.mean(alignment_scores)}')
    print(f'Average length-weighted alignment score: {np.mean(alignment_len_scores):.2f}')
    print(f'L2 distance for full alignment is {full_std_eval:.2f}, for match only {match_std_eval:.2f}')
    #print(f'Pearson r for full alignment to remora is {remora_full_correlation_eval}, for match only {remora_match_correlation_eval}')
    print(f'Pearson r for full alignment is {full_correlation_eval}, for match only {match_correlation_eval}')

def main(args):
    bam_index = BamIndex(args.bam, use_cached=False)

    full_events = load_full_events(args.full_events)

    naive_evaluation(full_events, bam_index)
    naive_expand_evaluation(full_events, bam_index)
    eval_chamfer(full_events, bam_index)

    alndf = pl.read_csv(args.alignments)

    aligned_event_evaluation(alndf)

DEFAULT_BAM  = '/mnt/sod2-project/csb4/wgs/metagenomics_data/projects/segmentation/segmentation_data/R10_Zymo_subsample/barcode24_zymo_wo_EC_1k_per_species_min_len_1k/refined_R10_zymo_wo_EC_segmenteval_subset.bam'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bam', type=str, default=DEFAULT_BAM, help="The path to the .bam file containing refined event borders stored under RR tag. This can be constructed following the ground truth pipeline")
    parser.add_argument('--full_events', type=str, default='full_info.csv', help="The path to a csv file generated from parquet file with full information on predicted events. The csv file can be constructed from parquet with convert_parquet_for_analysis.py")
    parser.add_argument('--alignments', type=str, default='alignments.csv', help="The path to aligned predicted and ground truth events obtained by running align_events.py")

    args = parser.parse_args()
    main(args)
