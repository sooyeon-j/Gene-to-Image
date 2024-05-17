import os, shutil, random
from pathlib import Path
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.utils.data
import pandas as pd
import glob


def set_seed(s, reproducible=False):
    "Set random seed for `random`, `torch`, and `numpy` (where available)"
    try:
        torch.manual_seed(s)
    except NameError:
        pass
    try:
        torch.cuda.manual_seed_all(s)
    except NameError:
        pass
    try:
        np.random.seed(s % (2**32 - 1))
    except NameError:
        pass
    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_mri(args):
    data_transforms = [
        T.Resize((args.height, args.width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ]
    data_transforms = [
        T.Resize((args.height, args.width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Lambda(lambda t: (t * 2) - 1),
    ]
    data_transform = T.Compose(data_transforms)
    #'/root/proj/dataset/img_all/gray'

    train_folder = torchvision.datasets.ImageFolder(
        root="/root/jsuyeon/dataset/img_with_gene/ADNI1_GO2/train",
        transform=data_transform,
    )
    val_folder = torchvision.datasets.ImageFolder(
        root="/root/jsuyeon/dataset/img_with_gene/ADNI1_GO2/val",
        transform=data_transform,
    )

    if args.test_balanced:
        test_folder = torchvision.datasets.ImageFolder(
            root="/root/jsuyeon/dataset/img_with_gene/ADNI1_GO2/test_balanced",
            transform=data_transform,
        )
        
        gen_folder = torchvision.datasets.ImageFolder(
            root="/root/jsuyeon/samples_v3_balance/{}".format(args.dataset),
            transform=data_transform,
        )
    else:
        test_folder = torchvision.datasets.ImageFolder(
            root="/root/jsuyeon/dataset/img_with_gene/ADNI1_GO2/test",
            transform=data_transform,
        )
        
        gen_folder = torchvision.datasets.ImageFolder(
            root="/root/jsuyeon/samples_v3/{}".format(args.dataset),
            transform=data_transform,
        )
            
    concat_dataset = torch.utils.data.ConcatDataset([train_folder, gen_folder])

    train_loader = DataLoader(train_folder, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_folder, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_folder, batch_size=args.batch_size, shuffle=True)
    concat_loader = DataLoader(concat_dataset, batch_size=args.batch_size, shuffle=True)
    gen_loader = DataLoader(gen_folder, batch_size=args.batch_size, shuffle=True)
    
    return train_loader, gen_loader, val_loader, test_loader, concat_loader


def one_batch(dl):
    return next(iter(dl))


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(
        torch.cat(
            [
                torch.cat([i for i in images.cpu()], dim=-1),
            ],
            dim=-2,
        )
        .permute(1, 2, 0)
        .cpu()
    )
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def mk_folders(run_name, num_classes):
    os.makedirs("models_bl", exist_ok=True)
    # os.makedirs("results_bl", exist_ok=True)
    os.makedirs(os.path.join("models_bl", run_name), exist_ok=True)
    # os.makedirs(os.path.join("results_bl", run_name), exist_ok=True)
    # for i in range(num_classes):
    # os.makedirs(os.path.join("results_bl", run_name, f"{i}"), exist_ok=True)
