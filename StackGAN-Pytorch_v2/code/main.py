from __future__ import print_function
import torch.backends.cudnn as cudnn
import torch
import torchvision.transforms as transforms
import time

import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil
import dateutil.tz
import wandb
from contextlib import nullcontext

from baseline.classifier import Classifier

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

from miscc.datasets import dataloader
from miscc.config import cfg, cfg_from_file
from miscc.utils import mkdir_p
from trainer import GANTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='birds_stage1.yml', type=str)
    parser.add_argument('--gpu',  dest='gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--run_name', default= '', type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    ts = time.strftime("%b%d-%H:%M:%S", time.gmtime())
    args = parse_args()
    
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    if args.run_name != '':
        cfg.RUN_NAME = args.run_name
    # print('Using config:')
    # pprint.pprint(cfg)
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    cfg.manualSeed = args.manualSeed
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)
    
    cfg.ts = ts
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
                 (cfg.DATASET_NAME, cfg.CONFIG_NAME, cfg.RUN_NAME)

    num_gpu = len(cfg.GPU_ID.split(','))
    train_loader, val_loader, test_loader = dataloader(cfg.DATA_DIR, input_size=cfg.IMSIZE, batch_size= cfg.TRAIN.BATCH_SIZE)
    
    with wandb.init(project= "MRI_GAN", group= "Stack-GAN",  config= cfg) if cfg.USE_WANDB else nullcontext():
        if cfg.TRAIN.FLAG == 'train':
            algo = GANTrainer(output_dir)
            algo.train(train_loader, cfg.STAGE)
        elif cfg.TRAIN.FLAG == 'sample':
            algo = GANTrainer(output_dir)
            algo.sample(test_loader, cfg.STAGE)
        elif cfg.TRAIN.FLAG == 'classify':
            classification = Classifier(cfg)
            classification.classify()
