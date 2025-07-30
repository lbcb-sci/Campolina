import argparse
import polars as pl
import pod5 as p5
import numpy as np

from tqdm import tqdm

def main(args):

    df = pl.read_parquet(args.parquet)

    # Build map of read_id → sorted event_start list
    borders_series = (
        df
        .group_by("read_id", maintain_order=True)
        .agg(pl.col("event_start"))  # becomes a list column
        .rename({"event_start": "borders"})
    )

    # Convert to Python dict for lookup
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
            signal = read.signal  # NumPy array
            borders = borders_map[read_id]

            if borders.is_empty(): continue

            # compute splits
            segments = np.split(signal, borders)[1:]

            for st, seg in zip(borders, segments):
                full_info.append((
                    read_id, 
                    int(st), 
                    len(seg), 
                    float(np.mean(seg)), 
                    float(np.std(seg))
                ))

    cols = {
        'read_id': pl.Categorical, 
        'event_start': pl.Int32, 
        'event_len': pl.Int32,
        'event_mean': pl.Float32, 
        'event_std': pl.Float32,
    }

    out = pl.DataFrame(full_info, schema=cols, orient="row")
    out.write_csv(args.target)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet', type=str, required=True, help='Path to parquet file with predicted borders')
    parser.add_argument('--pod5', type=str, required=True, help='Path to .pod5 with the corresponding signals')
    parser.add_argument('--target', type=str, required=True, help='Path to target csv file with full event info')

    main(parser.parse_args())