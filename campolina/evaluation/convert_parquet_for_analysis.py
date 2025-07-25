import argparse
from tqdm import tqdm
import polars as pl
import pod5 as p5
import numpy as np

def main(args):
    df = pl.read_parquet(args.parquet)

    cols = {
        'read_id': pl.Categorical,
        'event_start': pl.Int32,
        'event_len': pl.Int32,
        'event_mean': pl.Float32,
        'event_std': pl.Float32,
    }

    full_info = []

    with p5.Reader(args.pod5) as reader:
        for rid, g in tqdm(df.group_by('read_id')):

            for r in reader.reads(selection=rid, preload='samples', missing_ok=False): 
                signal = r.signal

            borders = g['event_start']
            signal_events = np.split(signal, borders)[1:]
            signal_peaks = borders
            full_info.extend([
                (rid, signal_peak, len(e), np.mean(e), np.std(e))
                for signal_peak, e in zip(signal_peaks, signal_events)
            ])

    full_info = pl.DataFrame(full_info, schema=cols, orient='row')
    full_info.write_csv(args.target)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet', type=str, required=True, help='Path to parquet file with predicted borders')
    parser.add_argument('--pod5', type=str, required=True, help='Path to .pod5 with the corresponding signals')
    parser.add_argument('--target', type=str, required=True, help='Path to target csv file with full event info')

    main(parser.parse_args())