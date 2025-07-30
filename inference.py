import os
os.environ['POLARS_MAX_THREADS'] = '32'
import polars as pl
pl.enable_string_cache()
pl.Config.set_fmt_str_lengths(38)

import argparse
import time
import tqdm
import torch
import multiprocessing as mp
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import pod5
from pathlib import Path

from campolina.data.utils import get_raw_batch3
from campolina.data.output_utils import process_output_format
from campolina.model import EventDetector, UNet
from campolina.data.pod5_util import (
    get_pod5_readid_pairs,
    comp_cumsum_gpu,
    comp_tstat_gpu,
    diff1_gpu,
    window_mean_std_gpu,
)

DEFAULT_POD5_DIR = '/mnt/sod2-project/csb4/wgs/metagenomics_data/projects/segmentation/segmentation_data/R10_Zymo_subsample/barcode24_zymo_wo_EC_1k_per_species_min_len_1k/'

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

def predict_detect(model: torch.nn.Module, batch, device):
    torch.cuda.synchronize()
    batch = batch[:, :4, :].to(device)
    logits = model(batch).squeeze().detach().cpu()
    torch.cuda.synchronize()
    return logits

def predict(
        model_path: str, 
        model: str,
        devices: list, 
        pod5_rids_pairs, 
        batch_size: int, 
        target_file: str, 
        mode: str,
    ):

    device = devices[0]

    # TODO why do we have to do that ?
    state_dict = {
        k.replace('_orig_mod.', ''): v 
        for k, v in torch.load(model_path, map_location=device, weights_only=True).items()
    }

    match model:
        case 'default':
            model = EventDetector(
                in_channels=4, 
                out_channels=[32, 64, 64, 128, 128],
                classification_head=[128, 1], 
                kernel_size_one=3, 
                kernel_size_all=31
            ).to(device)

        case 'unet':
            model = UNet.make_default().to(device)
        
        case _:
            print('model must be in [default, unet]')
            exit(0)

    print(model.load_state_dict(state_dict, strict=True))
    model = model.eval()

    # Generate schema
    schema = pa.schema([
        ('read_id', pa.string()),
        ('event_start', pa.int32())
    ])

    output_path = f"{target_file}.parquet"

    # Init worker processes
    queue = mp.Queue()
    process = mp.Process(target=writer_worker, args=(queue, output_path, schema, mode))
    process.start()

    for pod5_path, rids in pod5_rids_pairs:
        reader = pod5.Reader(pod5_path)

        for chunks, chunk_borders, read_ids, signal_chunks in tqdm.tqdm(get_raw_batch3(reader, rids, batch_size)):

            torch_chunks = torch.Tensor(np.array(chunks)).to(device)
            cumsum_sig_gpu, cumsum_sig_square_gpu = comp_cumsum_gpu(torch_chunks)
            tstat1_gpu = comp_tstat_gpu(cumsum_sig_gpu, cumsum_sig_square_gpu, 6000, 3)
            diff_gpu = diff1_gpu(torch_chunks)
            gpu_w_means, gpu_w_stds = window_mean_std_gpu(torch_chunks, wlen=3)

            signal = torch.stack([torch_chunks, diff_gpu, gpu_w_means, gpu_w_stds, tstat1_gpu], dim=1)
            logits = predict_detect(model, signal, device)

            queue.put((logits, chunk_borders, read_ids, signal_chunks))

    # Close workers
    queue.put(None)
    process.join()

def main(args):
    full_start = time.time()

    if args.gpu is not None and len(args.gpu) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join((str(x) for x in args.gpu))
        devices = [torch.device("cuda", x) for x in range(len(args.gpu))]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        devices = [torch.device("cpu")]

    pod5_readid_pairs = get_pod5_readid_pairs(args.pod5_dir)

    predict(
        model_path=args.model_path, 
        model=args.model,
        devices=devices, 
        pod5_rids_pairs=pod5_readid_pairs, 
        batch_size=args.batch_size, 
        target_file=f'{args.target_dir}/{args.output}', 
        mode=args.mode,
    )

    full_end = time.time()
    print(f'Full execution took {full_end - full_start}')


def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pod5_dir', type=Path, default=DEFAULT_POD5_DIR)
    parser.add_argument('--model_path', type=Path, default='default.pth')
    parser.add_argument('--model', choices=['default', 'unet'], default='default')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--gpu', default=[0])
    parser.add_argument('--mode', choices=['raw', 'analysis'], default='raw')
    parser.add_argument('--target_dir', type=Path, default='./')
    parser.add_argument('--output', type=str, default='inference_output')
    return parser

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    main(make_argparser().parse_args())
