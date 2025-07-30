import argparse
import polars as pl
import pod5 as p5
import numpy as np
from tqdm import tqdm

def main(args):
    # build map of read_id → sorted event_start list
    borders_series = (
        pl.read_parquet(args.parquet)
        .group_by("read_id", maintain_order=True)
        .agg(pl.col("event_start"))  # becomes a list column
        .rename({"event_start": "borders"})
    )

    # convert to python dict for lookup
    borders_map = {
        read_id: borders
        for read_id, borders in zip(borders_series["read_id"], borders_series["borders"])
    }

    full_info = []
    with p5.Reader(args.pod5) as reader:

        reads = reader.reads(
            selection=borders_map.keys(), 
            preload="samples",
        )

        for i, read in enumerate(tqdm(reads)):
            if i == 100: break

            read_id = str(read.read_id)
            signal = read.signal # numpy array
            borders = borders_map[read_id]

            if borders.is_empty(): continue

            # compute splits
            segments = np.split(signal, borders)[1:]

            for st, seg in zip(borders, segments):
                data = [read_id, int(st), 0]
                data.extend([np.nan, np.nan] if len(seg) == 0 else [np.mean(seg), np.std(seg)])
                full_info.append(tuple(data))

    cols = {
        'read_id': pl.Categorical, 
        'event_start': pl.Int32, 
        'event_len': pl.Int32,
        'event_mean': pl.Float32, 
        'event_std': pl.Float32,
    }

    # write output
    pl.DataFrame(full_info, schema=cols, orient="row").write_csv(args.target)

DEFAULT_POD5 = '/mnt/sod2-project/csb4/wgs/metagenomics_data/projects/segmentation/segmentation_data/R10_Zymo_subsample/barcode24_zymo_wo_EC_1k_per_species_min_len_1k/barcode24_zymo_subsampled_wo_EC_min_len_1k.pod5'

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet', type=str, default='inference_output.parquet', help='Path to parquet file with predicted borders')
    parser.add_argument('--pod5', type=str, default=DEFAULT_POD5, help='Path to .pod5 with the corresponding signals')
    parser.add_argument('--target', type=str, default='full_info.csv', help='Path to target csv file with full event info')
    return parser

if __name__ == '__main__':
    main(make_argparser().parse_args())