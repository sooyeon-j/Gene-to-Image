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
# from monai.data import DataLoader, Dataset

def get_mri(args, dataset_path, snp_path=None, type='ae'):
    # data_transforms = [T.Resize((args.data.height, args.data.width)), T.RandomHorizontalFlip(), T.ToTensor(), T.Lambda(lambda t: (t * 2) - 1)]
    data_transforms = [T.Resize((args.data.height, args.data.width)), T.RandomHorizontalFlip(), T.ToTensor()]    
    data_transform = T.Compose(data_transforms)
    
    if type == 'ae':   
        image_folder = torchvision.datasets.ImageFolder(root = dataset_path, transform = data_transform)
        train_size = int(0.7 * len(image_folder))
        val_size = (len(image_folder) - train_size) // 2
        test_size = len(image_folder) - train_size - val_size

        train_dataset, test_dataset = torch.utils.data.random_split(image_folder, [train_size + val_size, test_size])
        train_loader = DataLoader(train_dataset, num_workers= 20, batch_size = args.data.ae_batch_size, shuffle = True)
        test_loader = DataLoader(test_dataset, num_workers= 20, batch_size = args.data.ae_batch_size, shuffle = False)
        return train_loader, test_loader
    
    # indices = list(range(len(image_folder)))
    # np.random.shuffle(indices)
    # train_indices, test_indices = indices[:(train_size + val_size)], indices[(train_size + val_size):]
    # train_indices, val_indices = indices[:train_size], indices[train_size:]
    
    train_image_folder = torchvision.datasets.ImageFolder(root = os.path.join(dataset_path, 'train'), transform = data_transform)
    val_image_folder = torchvision.datasets.ImageFolder(root = os.path.join(dataset_path, 'val'), transform = data_transform)
    test_image_folder = torchvision.datasets.ImageFolder(root = os.path.join(dataset_path, 'test'), transform = data_transform)
    
    image_folder_list = [train_image_folder, val_image_folder, test_image_folder]
    snp = pd.read_csv(snp_path)
    ds = []
    
    for image_folder_ in image_folder_list:
        img_list = [image_folder_[i][0] for i in range(len(image_folder_))]
        # label_list = [torch.tensor(image_folder_[i][1]) for i in range(len(image_folder_))]
        id_list = [image_folder_.samples[i][0][-14:-4] for i in range(len(image_folder_))]
        label_list = []
        snp_list = []

        for i in range(len(id_list)):
            snp_ = snp[snp['PTID'] == id_list[i]]
            label = torch.tensor(snp_.iloc[:, -1].item())
            snp_ = snp_.iloc[:, 1:-2].to_numpy()
            label_list.append(label)
            snp_list.append(snp_.reshape(snp_.shape[1]))

        # if image_folder_ == train_image_folder:
        #     num_to_select = int(len(label_list) * 0.2)
        #     list_of_random_idx = random.sample(range(len(label_list)), num_to_select)
        #     for j in list_of_random_idx:
        #         label_list[j] = torch.tensor(3)
        #         snp_list[j] = -1 * np.ones((snp_.shape[1],))
       
        img_tensor = torch.stack(img_list)
        label_tensor = torch.stack(label_list)
        snp_np = np.array(snp_list)
        # snp_tensor = torch.tensor(snp_np, dtype = torch.float32)
        snp_tensor = torch.tensor(snp_np, dtype = torch.long)
        ds.append(TensorDataset(img_tensor, label_tensor, snp_tensor))

    # train_ds = TensorDataset(img_tensor[train_indices], label_tensor[train_indices], snp_tensor[train_indices])
    # val_ds = TensorDataset(img_tensor[val_indices], label_tensor[val_indices], snp_tensor[val_indices])
    # test_ds = TensorDataset(img_tensor[test_indices], label_tensor[test_indices], snp_tensor[test_indices])

    train_loader = DataLoader(ds[0], num_workers= 20, batch_size = args.data.dm_batch_size, shuffle = True)
    val_loader = DataLoader(ds[1], num_workers= 20, batch_size = 10, shuffle = True)
    test_loader = DataLoader(ds[2], num_workers= 20, batch_size = args.sample.batch_size, shuffle = True)
    
    if type == 'sample':
        return test_loader

    return train_loader, val_loader, test_loader
        

def one_batch(dl):
    return next(iter(dl))


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()

def save_images(images, path):
    grid = torchvision.utils.make_grid(images)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    plt.imsave(path, ndarr)
    # ndarr = 255. * rearrange(images.cpu().numpy(), 'c h w -> h w c')
    # im = Image.fromarray(ndarr.astype(np.uint8))
    # im.save(path)

def mk_folders(config, train=True):
    if train:
        os.makedirs("results", exist_ok=True)
        os.makedirs(os.path.join("results", config.run_name), exist_ok=True)
        # os.makedirs(os.path.join("results", config.run_name, "AE"), exist_ok=True)
        os.makedirs(os.path.join("results", config.run_name, "diffusion"), exist_ok=True)
    
    else:
        os.makedirs(os.path.join("results", config.run_name, "samples"), exist_ok=True)
        os.makedirs(os.path.join("results", config.run_name, "samples", config.ts), exist_ok=True)
        os.makedirs(os.path.join("results", config.run_name, "samples", config.ts, "AD"), exist_ok=True)
        os.makedirs(os.path.join("results", config.run_name, "samples", config.ts, "MCI"), exist_ok=True)
        os.makedirs(os.path.join("results", config.run_name, "samples", config.ts, "CN"), exist_ok=True)
    