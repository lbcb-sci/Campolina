import argparse
import os
import tqdm

from pathlib import Path
from collections import Counter
import multiprocessing as mp

import torch
import numpy as np
from polars.polars import first
import torch.profiler
import torch_tensorrt
from torch.utils.data import DataLoader

os.environ['POLARS_MAX_THREADS'] = '32'

import polars as pl
pl.enable_string_cache(enable=True)
pl.Config.set_fmt_str_lengths(38)
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

from data.pod5_util import *
from data.utils import *
from data.output_utils import *
from data.loader_utils import *
from model.model import *
from model.tcn_model import TCNEventDetector

mp.set_start_method('spawn', force=True)

import time

log = False
torch.cuda.set_device(5)
target_device = torch_tensorrt.Device(gpu_id=5)

import torch._dynamo
torch._dynamo.config.suppress_errors = True

def find_positive_indices(row):
    return (row > 0).nonzero(as_tuple=True)[0]


def predict_detect(model, batch, device):
    prediction_start = time.time()
    with torch.autocast(device_type="cuda"), torch.no_grad():
        logits = model(torch.Tensor(batch).to(device, non_blocking=True)).squeeze()
    #logits = model(batch).squeeze()
    prediction_end = time.time()
    if log:
        print(f'Prediction took {prediction_end - prediction_start}')
    border_start = time.time()
    borders = [(logit > 0).nonzero(as_tuple=True)[0] for logit in logits]
    border_end = time.time()
    if log:
        print(f'Predictions->peaks took {border_end - border_start}')

    return borders


def predict(model_path, devices, pod5_rids_pairs, bs, tgt_file, workers, mode):
    print(f'Devices: {devices}')
    state_dict = torch.load(model_path, map_location=devices[0])
    model = EventDetector(in_channels=5, out_channels=[32, 64, 128, 256, 1024, 2048],
                          classification_head=[2048, 512, 32, 1], kernel_size_one=3, kernel_size_all=9).to(devices[0])
    #model = TCNEventDetector(in_channels = 5, channels=[32, 128, 256, 1024, 2048], kernel_size=3, classification_head=[2048, 256, 32, 1], dropout=0.1, causal=False, use_norm='batch_norm', activation='gelu').to(devices[0])

    model.load_state_dict(state_dict, strict=True)
    model.half()
    model = torch.compile(model, backend='torch_tensorrt', dynamic=False, fullgraph=True, options={"truncate_long_and_double": True, "enabled_precisions": {torch.float,torch.half}})
    model.eval()

    first_write=True

    if mode == 'raw':
        cols = ['read_id', 'event_start']
        writer = None
    else:
        cols = {'read_id': pl.Categorical, 'event_start': pl.Int32, 'event_len': pl.Int32, 'event_mean': pl.Float32,
                'event_std': pl.Float32}
        writer = open(f'{tgt_file}.csv', 'a')

    """queue = mp.Queue()
    reader = mp.Process(target=reader_worker, args=(queue, pod5_dir, workers))
    reader.start()

    dataset = SignalDataset(queue)
    loader = DataLoader(dataset, num_workers=workers)"""

    print(model)

    for pod5_path, rids in pod5_rids_pairs:
        reader = p5.Reader(pod5_path)

        for chunks, chunk_borders, read_ids, signal_chunks in tqdm.tqdm(get_raw_batch3(reader, rids, bs)):
            torch_chunks = torch.Tensor(np.array(chunks)).half().to(devices[0], non_blocking=True)
            #torch_chunks = torch.stack(chunks).to(devices[0], non_blocking=True)
            cumsum_sig_gpu, cumsum_sig_square_gpu = comp_cumsum_gpu(torch_chunks)
            tstat1_gpu = comp_tstat_gpu(cumsum_sig_gpu, cumsum_sig_square_gpu, 6000, 3)
            diff_gpu = diff1_gpu(torch_chunks)
            gpu_w_means, gpu_w_stds = window_mean_std_gpu(torch_chunks, wlen=3)
            signal = torch.stack([torch_chunks, diff_gpu, gpu_w_means, gpu_w_stds, tstat1_gpu], dim=1)
            #signal = torch_chunks.unsqueeze(1)

            peaks_start = time.time()
            peaks = predict_detect(model, signal, devices[0])
            peaks_end = time.time()
            if log:
                print(f'Generating peaks took {peaks_end - peaks_start}')

            output_start = time.time()
            events = process_output_format(peaks, chunk_borders, read_ids, mode, signal_chunks)
            output_end = time.time()
            if log:
                print(f'Output processing took {output_end - output_start}')

            if mode == 'analysis':
                events.collect().write_csv(writer, has_header=first_write)
                first_write = False
            else:
                table = pa.Table.from_pandas(pd.DataFrame(events, columns=cols))
                if writer is None:
                    writer = pq.ParquetWriter(f'{tgt_file}.parquet', table.schema)
                writer.write_table(table)

    if mode == 'raw':
        if writer is not None:
            writer.close()


def main(args):
    full_start = time.time()
    if args.gpu is not None and len(args.gpu) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join((str(x) for x in args.gpu))
        devices = [torch.device("cuda", x) for x in range(len(args.gpu))]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        devices = [torch.device("cpu")]

    #devices = ['cuda:5']
    pod5_readid_pairs = get_pod5_readid_pairs(args.pod5_dir)
    predict(args.model_path, devices, pod5_readid_pairs, args.bs,
                       f'{args.tgt_dir}/{args.abbrev}_events', args.workers, args.mode)
    full_end = time.time()
    if log:
        print(f'Full execution took {full_end - full_start}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pod5_dir', type=Path,
                        default='/home/bakics/scratch/signal_segmentation/Campolina_paper_predictions/R10_zymo/')
    parser.add_argument('--model_path', type=Path,
                        default='14112024_Focal_alpha0_8_gamma1_alpha5000_beta0_05_eta10_5channel_final1024_400bps_model.pth')
    parser.add_argument('--tgt_dir', type=Path,
                        default='/home/bakics/scratch/signal_segmentation/Campolina_paper_predictions/R10_zymo/resource_utilization_analysis/')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--gpu', default=['5'])
    parser.add_argument('--abbrev', type=str, default='test_multiread_multibatch')
    parser.add_argument('--delete_src', action='store_true', default=False)
    parser.add_argument('--mode', choices=['raw', 'analysis'], default='raw')
    parser.add_argument('--log_time', action='store_true')

    main(parser.parse_args())
