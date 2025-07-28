import argparse
import os
import time
import tqdm
from pathlib import Path
import multiprocessing as mp
import torch
import numpy as np

os.environ['POLARS_MAX_THREADS'] = '32'
import polars as pl
pl.enable_string_cache()
pl.Config.set_fmt_str_lengths(38)

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

from campolina.data.pod5_util import *
from campolina.data.utils import *
from campolina.data.output_utils import *
from campolina.data.loader_utils import *
from campolina.model import EventDetector, UNet

mp.set_start_method('spawn', force=True)
log=True

def find_positive_indices(row): return (row > 0).nonzero(as_tuple=True)[0]

def writer_worker(queue, output_path, schema, mode):
    writer = pq.ParquetWriter(output_path, schema)
    while True:
        item = queue.get()

        if item is None: break

        logits, chunk_borders, read_ids, signal_chunks = item
        peaks = [(logit > 0).nonzero(as_tuple=True)[0] for logit in logits]
        events = process_output_format(peaks, chunk_borders, read_ids, mode, signal_chunks)
        df = pd.DataFrame(events)
        table = pa.Table.from_pandas(df, schema=schema)
        writer.write_table(table)

    writer.close()

def predict_detect(model, batch, device):
    torch.cuda.synchronize()
    logits = model(torch.Tensor(batch).to(device)).squeeze().detach().cpu()
    torch.cuda.synchronize()
    return logits

def predict(model_path, devices, pod5_rids_pairs, bs, tgt_file, workers, mode):
    print(f'Devices: {devices}')
    device = devices[0]
    state_dict = torch.load(model_path, map_location=device)

    #model = EventDetector(in_channels=5, out_channels=[32, 64, 64, 128, 128],
                          #classification_head=[128, 1], kernel_size_one=3, kernel_size_all=31).to(devices[0])

    model = UNet.make_default().to(device)
    model.load_state_dict(state_dict, strict=False)#, strict=True)
    model.eval()

    # Generate schema
    schema = pa.schema([
        ('read_id', pa.string()),
        ('event_start', pa.int32())
    ])

    output_path = f"{tgt_file}.parquet"

    # Init worker processes
    queue = mp.Queue()
    process = mp.Process(target=writer_worker, args=(queue, output_path, schema, mode))
    process.start()

    batch_id = 0

    for pod5_path, rids in pod5_rids_pairs:
        reader = p5.Reader(pod5_path)
        for chunks, chunk_borders, read_ids, signal_chunks in tqdm.tqdm(get_raw_batch3(reader, rids, bs)):
            torch_chunks = torch.Tensor(np.array(chunks)).to(devices[0])
            cumsum_sig_gpu, cumsum_sig_square_gpu = comp_cumsum_gpu(torch_chunks)
            tstat1_gpu = comp_tstat_gpu(cumsum_sig_gpu, cumsum_sig_square_gpu, 6000, 3)
            diff_gpu = diff1_gpu(torch_chunks)
            gpu_w_means, gpu_w_stds = window_mean_std_gpu(torch_chunks, wlen=3)
            signal = torch.stack([torch_chunks, diff_gpu, gpu_w_means, gpu_w_stds, tstat1_gpu], dim=1)

            logits = predict_detect(model, signal, device)
            queue.put((logits, chunk_borders, read_ids, signal_chunks))
            batch_id += 1

    # Close workers
    queue.put(None)
    process.join()

    # Merge final output
    #print("Merging all output parquet files...")
    #merge_start = time.time()
    """all_tables = []
    for i in range(workers):
        path = os.path.join(output_dir, f"worker_{i}.parquet")
        all_tables.append(pq.read_table(path))
    merged_table = pa.concat_tables(all_tables)
    pq.write_table(merged_table, f"{tgt_file}.parquet")
    print(f"Final merged parquet written to {tgt_file}.parquet")"""

    """output_path = f"{tgt_file}.parquet"
    for i in range(workers):
        part_path = os.path.join(output_dir, f"worker_{i}.parquet")
        table = pq.read_table(part_path)
        if i == 0:
            pqwriter = pq.ParquetWriter(output_path, schema=schema)
            #pq.write_table(table, output_path)
        pqwriter.write(table)
        #else:
            #pq.write_table(table, output_path, append=True)"""

def main(args):
    full_start = time.time()
    if args.gpu is not None and len(args.gpu) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join((str(x) for x in args.gpu))
        devices = [torch.device("cuda", x) for x in range(len(args.gpu))]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        devices = [torch.device("cpu")]

    pod5_readid_pairs = get_pod5_readid_pairs(args.pod5_dir)
    predict(args.model_path, devices, pod5_readid_pairs, args.bs,
                       f'{args.tgt_dir}/{args.abbrev}_events', 1, args.mode)
    full_end = time.time()
    if log: print(f'Full execution took {full_end - full_start}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--pod5_dir', type=Path,
                        default='/mnt/sod2-project/csb4/wgs/metagenomics_data/projects/segmentation/segmentation_data/R10_Zymo_subsample/barcode24_zymo_wo_EC_1k_per_species_min_len_1k/')

    parser.add_argument('--model_path', type=Path,
                        default='/home/stumaxime/scratch/campolina/models/unet_1.5M/epoch[1]_step[200]_loss[311.1089].pth')

    parser.add_argument('--tgt_dir', type=Path, default='./')

    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--bs', type=int, default=1024)
    parser.add_argument('--gpu', default=[1])
    parser.add_argument('--abbrev', type=str, default='output')
    parser.add_argument('--delete_src', action='store_true', default=False)
    parser.add_argument('--mode', choices=['raw', 'analysis'], default='raw')
    parser.add_argument('--log_time', action='store_true')

    main(parser.parse_args())
