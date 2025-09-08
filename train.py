# This script simply parses the provided train_config file and calls `campolina.train()`.

import argparse
import warnings
import logging
import json
import os

import torch
import campolina 

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='train_config.json')
    parser.add_argument('--run_name', default='campolina')

    args = parser.parse_args()
    with open(args.config_file, 'r') as inf: scope = json.load(inf)

    if 'gpu' in scope and len(scope['gpu']) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join((str(x) for x in scope['gpu']))
        scope["devices"] = [torch.device("cuda", x) for x in range(len(scope['gpu']))]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        scope["devices"] = [torch.device("cpu")]

    logging.basicConfig(level=logging.INFO)
    logging.info(f'using {scope["devices"]}')

    campolina.train(scope=scope, run_name=args.run_name) 