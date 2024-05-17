from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os, shutil, random
from pathlib import Path
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data
import pandas as pd
import random
from einops import rearrange
from monai import transforms

from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from miscc.config import cfg

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random


def prepare_mri_data(data):
    imgs, captions, captions_lens, class_ids, keys = data
    im_size = [cfg.TREE.BASE_SIZE * (2**x) for x in range(cfg.TREE.BRANCH_NUM)]
    imgs = [transforms.Resize(size)(imgs) for size in im_size]
    captions = captions.permute(0, 2, 1)
    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = torch.sort(captions_lens, 0, True)

    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))

    captions = captions[sorted_cap_indices].squeeze()
    class_ids = class_ids[sorted_cap_indices].numpy()
    # sent_indices = sent_indices[sorted_cap_indices]
    keys = [keys[i] for i in sorted_cap_indices.numpy()]
    # print('keys', type(keys), keys[-1])  # list
    if cfg.CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)

    return [real_imgs, captions, sorted_cap_lens, class_ids, keys]


def get_mri(
    # args,
    BATCH_SIZE=32,
    dataset_path=None,
    snp_path="/root/jsuyeon/yujees/AttnGAN/data/dataset/adni1_and_2_qc_dropna_with_apoe_adnimerge_mri.csv",
    type="nope",
):
    # data_transforms = [T.Resize((args.data.height, args.data.width)), T.RandomHorizontalFlip(), T.ToTensor(), T.Lambda(lambda t: (t * 2) - 1)]
    if cfg.TREE.BASE_SIZE:
        img_size = cfg.TREE.BASE_SIZE
    else:
        img_size = 256
    data_transforms = [
        T.Resize((img_size, img_size)),
        # T.Resize((args.data.height, args.data.width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    data_transform = T.Compose(data_transforms)

    # indices = list(range(len(image_folder)))
    # np.random.shuffle(indices)
    # train_indices, test_indices = indices[:(train_size + val_size)], indices[(train_size + val_size):]
    # train_indices, val_indices = indices[:train_size], indices[train_size:]

    dataset_path = (
        "/root/jsuyeon/yujees/AttnGAN/data/dataset/img_with_gene/ADNI1_GO2/gray"
    )

    train_image_folder = torchvision.datasets.ImageFolder(
        root=os.path.join(dataset_path, "train"), transform=data_transform
    )
    val_image_folder = torchvision.datasets.ImageFolder(
        root=os.path.join(dataset_path, "val"), transform=data_transform
    )
    test_image_folder = torchvision.datasets.ImageFolder(
        root=os.path.join(dataset_path, "test"), transform=data_transform
    )

    image_folder_list = [train_image_folder, val_image_folder, test_image_folder]
    snp = pd.read_csv(snp_path)
    ds = []

    for image_folder_ in image_folder_list:
        img_list = [image_folder_[i][0] for i in range(len(image_folder_))]
        # label_list = [torch.tensor(image_folder_[i][1]) for i in range(len(image_folder_))]
        id_list = [
            image_folder_.samples[i][0][-14:-4] for i in range(len(image_folder_))
        ]
        label_list = []
        snp_list = []

        for i in range(len(id_list)):
            snp_ = snp[snp["PTID"] == id_list[i]]
            label = torch.tensor(snp_.iloc[:, -1].item())
            snp_ = snp_.iloc[:, 1:-2].to_numpy()
            label_list.append(label)
            snp_list.append(snp_.reshape(snp_.shape[1]))

        # base_size = cfg.TREE.BASE_SIZE
        # for i in range(cfg.TREE.BRANCH_NUM):
        #     base_size = base_size * 2

        # if image_folder_ == train_image_folder:
        #     num_to_select = int(len(label_list) * 0.2)
        #     list_of_random_idx = random.sample(range(len(label_list)), num_to_select)
        #     for j in list_of_random_idx:
        #         label_list[j] = torch.tensor(3)
        #         snp_list[j] = -1 * np.ones((snp_.shape[1],))

        img_tensor = torch.stack(img_list)
        label_tensor = torch.stack(label_list)
        snp_np = np.array(snp_list)
        snp_tensor = torch.tensor(snp_np, dtype=torch.float32).unsqueeze(1)
        ds.append(
            TensorDataset(
                img_tensor,
                snp_tensor,
                torch.ones(len(snp_tensor)) * 50,
                label_tensor,
                torch.zeros(len(snp_tensor), 1, 1),
            )
        )

    # train_ds = TensorDataset(img_tensor[train_indices], label_tensor[train_indices], snp_tensor[train_indices])
    # val_ds = TensorDataset(img_tensor[val_indices], label_tensor[val_indices], snp_tensor[val_indices])
    # test_ds = TensorDataset(img_tensor[test_indices], label_tensor[test_indices], snp_tensor[test_indices])
    return ds[0], ds[1], ds[2]

    train_loader = DataLoader(
        ds[0],
        num_workers=20,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(ds[1], num_workers=20, batch_size=10, shuffle=True)
    test_loader = DataLoader(
        ds[2], num_workers=20, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    if type == "sample":
        return test_loader

    return train_loader, val_loader, test_loader
