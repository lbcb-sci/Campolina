import argparse
import os
import pickle

import numpy as np
import pod5
from tqdm import tqdm

import multiprocessing as mp
from collections import Counter

from bam_util import BamIndex
from pod5_util import get_reads, process_chunk


def merge_npy(chunk_paths, label_paths, tgt_dir, save_src):     #TODO extremely slow, np.concatenate should be faster I guess
    with open(f'{tgt_dir}/chunks.npy', 'wb') as chunkf, open(f'{tgt_dir}/labels.npy', 'wb') as labelf:
        for cf, lf in zip(chunk_paths, label_paths):
            tqdm.write(f'Appending data from {cf.split("/")[-1]} and {lf.split("/")[-1]}')
            with open(cf, 'rb') as incf, open(lf, 'rb') as inlf:
                while True:
                    try:
                        np.save(chunkf, np.load(incf))
                        np.save(labelf, np.load(inlf))
                    except:
                        print(f'Finished writing')
                        break
            if not save_src:
                os.remove(cf)
                os.remove(lf)


def concat_npy(chunk_paths, label_paths, tgt_dir, save_src):
    chunk_arrays = []
    label_arrays = []
    for cf, lf in zip(chunk_paths, label_paths):
        tqdm.write(f'Appending data from {cf.split("/")[-1]} and {lf.split("/")[-1]}')
        with open(cf, 'rb') as incf, open(lf, 'rb') as inlf:
            while True:
                try:
                    chunk_arrays.append(np.load(incf))
                    label_arrays.append(np.load(inlf))
                except:
                    break
        if not save_src:
            os.remove(cf)
            os.remove(lf)

    with open(f'{tgt_dir}/chunks.npy', 'wb') as chunkf, open(f'{tgt_dir}/labels.npy', 'wb') as labelf:
        for c, l in tqdm(zip(chunk_arrays, label_arrays)):
            np.save(chunkf, c)
            np.save(labelf, l)


def pickle_npy(chunk_paths, label_paths, tgt_dir, bs, save_src):
    chunk_arrays = []
    label_arrays = []
    for cf, lf in zip(chunk_paths, label_paths):
        tqdm.write(f'Appending data from {cf.split("/")[-1]} and {lf.split("/")[-1]}')
        with open(cf, 'rb') as incf, open(lf, 'rb') as inlf:
            while True:
                try:
                    chunk_arrays.append(np.load(incf))
                    label_arrays.append(np.load(inlf))
                except:
                    break
        if not save_src:
            os.remove(cf)
            os.remove(lf)

    chunk_arrays = np.split(chunk_arrays, range(bs, len(chunk_arrays), bs))
    label_arrays = np.split(label_arrays, range(bs, len(label_arrays), bs))

    tqdm.write('Saving chunk and label batches')
    with open(f'{tgt_dir}/chunks.npy', 'wb') as chunkf, open(f'{tgt_dir}/labels.npy', 'wb') as labelf:
        pickle.dump(chunk_arrays, chunkf, pickle.HIGHEST_PROTOCOL)
        pickle.dump(label_arrays, labelf, pickle.HIGHEST_PROTOCOL)


def batch_npy(chunk_paths, label_paths, tgt_dir, bs, save_src):
    chunk_arrays = []
    label_arrays = []
    for cf, lf in zip(chunk_paths, label_paths):
        tqdm.write(f'Appending data from {cf.split("/")[-1]} and {lf.split("/")[-1]}')
        with open(cf, 'rb') as incf, open(lf, 'rb') as inlf:
            while True:
                try:
                    chunk_arrays.append(np.load(incf))
                    label_arrays.append(np.load(inlf))
                except:
                    break
        if not save_src:
            os.remove(cf)
            os.remove(lf)

    chunk_arrays = np.split(chunk_arrays, range(bs, len(chunk_arrays), bs))
    label_arrays = np.split(label_arrays, range(bs, len(label_arrays), bs))

    tqdm.write('Saving chunk and label batches')
    with open(f'{tgt_dir}/chunks.npy', 'wb') as chunkf, open(f'{tgt_dir}/labels.npy', 'wb') as labelf:
        for chunkb, labelb in tqdm(zip(chunk_arrays, label_arrays)):
            np.save(chunkf, chunkb)
            np.save(labelf, labelb)


def extract_worker(bam_idx, pod5_f, chunk_file, label_file, in_queue, out_queue):
    with open(f'{chunk_file}', 'wb') as chunkf, open(f'{label_file}', 'wb') as labelf:
        while (read_ids := in_queue.get()) is not None:
            status_count = 0
            for read in get_reads(pod5_f, read_ids):
                for alignment in bam_idx.get_alignment(str(read.read_id)):
                    if alignment is None:
                        continue
                    signal_chunks, chunk_borders = process_chunk(aln=alignment, read=read)
                    if signal_chunks is None:
                        #tqdm.write(f'Could not extract info for read {read.read_id}')
                        continue
                    for i in range(len(signal_chunks)):
                        np.save(chunkf, signal_chunks[i])
                        np.save(labelf, chunk_borders[i].astype(np.uint8))
                        status_count += 1
            out_queue.put(status_count)


def main_parallel(args):
    if not os.path.exists(args.tgt_dir):
        os.mkdir(args.tgt_dir)

    with pod5.Reader(args.pod5) as pf:
        read_ids = pf.read_ids
    #per_thread = (len(read_ids) // args.threads) // 10
    read_ids_list = np.split(np.array(read_ids), range(100, len(read_ids), 100))
    in_queue = mp.Queue()

    for r_id in read_ids_list:
        in_queue.put(r_id)
    for _ in range(args.threads):
        in_queue.put(None)

    out_queue = mp.Queue()
    workers = [None] * args.threads
    chunks_path = [None] * args.threads
    labels_path = [None] * args.threads

    bam_idx = BamIndex(args.bam)

    for i in range(args.threads):
        chunks_path[i] = f'{args.tgt_dir}/chunks_{i}.npy'
        labels_path[i] = f'{args.tgt_dir}/labels_{i}.npy'

        workers[i] = mp.Process(target=extract_worker,
                                args=(bam_idx, args.pod5, chunks_path[i], labels_path[i], in_queue, out_queue),
                                daemon=True)
        workers[i].start()

    pbar = tqdm(total=len(read_ids_list))
    status_count = Counter({'P': 0})
    while pbar.n < len(read_ids_list):
        status = out_queue.get()
        status_count['P'] += status

        pbar.set_postfix(status_count, refresh=False)
        pbar.update()

    for w in workers:
        w.join()

    tqdm.write('Batch extraction final step')
    batch_npy(chunks_path, labels_path, args.tgt_dir, args.bs, args.save_src)

def main(args):
    bam_idx = BamIndex(args.bam)

    if not os.path.exists(args.tgt_dir):
        os.mkdir(args.tgt_dir)

    with open(f'{args.tgt_dir}/chunks.npy', 'wb') as chunkf, open(f'{args.tgt_dir}/labels.npy', 'wb') as labelf:
        for read in tqdm(get_reads(args.pod5)):
            for alignment in bam_idx.get_alignment(str(read.read_id)):
                if alignment is None:
                    continue
                signal_chunks, chunk_borders = process_chunk(aln=alignment, read=read)
                if signal_chunks is None:
                    tqdm.write(f'Could not extract info for read {read.read_id}')
                for i in range(len(signal_chunks)):
                    np.save(chunkf, signal_chunks[i])
                    np.save(labelf, chunk_borders[i].as_type(np.uint8))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bam', default='../../samples/detecting_dir/sample_EC_chrom_r104_dorado_refined.bam')
    parser.add_argument('--pod5', default='../../samples/detecting_dir/testsample_EC_chromosome.pod5')
    parser.add_argument('--tgt_dir', default='../../samples/detecting_dir/test_tgt_dir')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--save_src', action='store_true', default=False)

    main_parallel(parser.parse_args())