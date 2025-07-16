import argparse
import json
import os
import warnings
import logging
import torch

from campolina import train 

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    torch.manual_seed(12345)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='train_config.json')

    args = parser.parse_args()
    with open(args.config_file, 'r') as inf: scope = json.load(inf)

    if 'gpu' in scope and len(scope['gpu']) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join((str(x) for x in scope['gpu']))
        #scope["devices"] = [torch.device("cuda", x) for x in range(len(scope['gpu']))]
        scope["devices"] = [torch.device("cuda", x) for x in range(len(scope['gpu']))]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        scope["devices"] = [torch.device("cpu")]

    logging.basicConfig(level=logging.INFO)
    logging.info(f'using {scope["devices"]}')

    train(scope) # main function