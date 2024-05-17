import torch
import os
import torchvision
import numpy as np

from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data
import pandas as pd
import torchvision.transforms as T


def dataloader(input_size, batch_size, split='train'):
    data_transforms = [T.Resize((input_size[0], input_size[1])), T.RandomHorizontalFlip(), T.ToTensor(),
                       T.Lambda(lambda t: (t * 2) - 1)]
                    # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    data_transform = T.Compose(data_transforms)

    image_folder = torchvision.datasets.ImageFolder(root = '/root/proj/dataset/img_with_gene/ADNI1_GO2/gray', transform = data_transform)
    # train_size = int(1. * len(image_folder))
    # val_size = (len(image_folder) - train_size) // 2
    # test_size = len(image_folder) - train_size - val_size
    
    train_image_folder = torchvision.datasets.ImageFolder(root = os.path.join('/root/proj/dataset/img_with_gene/ADNI1_GO2/gray', 'train'), transform = data_transform)
    val_image_folder = torchvision.datasets.ImageFolder(root = os.path.join('/root/proj/dataset/img_with_gene/ADNI1_GO2/gray', 'val'), transform = data_transform)
    test_image_folder = torchvision.datasets.ImageFolder(root = os.path.join('/root/proj/dataset/img_with_gene/ADNI1_GO2/gray', 'test'), transform = data_transform)
    
    image_folder_list = [train_image_folder, val_image_folder, test_image_folder]
    snp = pd.read_csv('/root/proj/dataset/adni1_and_2_qc_dropna_with_apoe_adnimerge_mri.csv')
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
        label_tensor = torch.stack(label_list) # 1
        snp_np = np.array(snp_list)
        snp_tensor = torch.tensor(snp_np, dtype = torch.float32) # 50
        ds.append(TensorDataset(img_tensor, label_tensor, snp_tensor))

    # train_ds = TensorDataset(img_tensor[train_indices], label_tensor[train_indices], snp_tensor[train_indices])
    # val_ds = TensorDataset(img_tensor[val_indices], label_tensor[val_indices], snp_tensor[val_indices])
    # test_ds = TensorDataset(img_tensor[test_indices], label_tensor[test_indices], snp_tensor[test_indices])

    train_loader = DataLoader(ds[0], num_workers= 20, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(ds[1], num_workers= 20, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(ds[2], num_workers= 20, batch_size = batch_size, shuffle = False)

    return train_loader, val_loader, test_loader

    # transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    # if dataset == 'mnist':
    #     data_loader = DataLoader(
    #         datasets.MNIST('data/mnist', train=True, download=True, transform=transform),
    #         batch_size=batch_size, shuffle=True)
    # elif dataset == 'fashion-mnist':
    #     data_loader = DataLoader(
    #         datasets.FashionMNIST('data/fashion-mnist', train=True, download=True, transform=transform),
    #         batch_size=batch_size, shuffle=True)
    # elif dataset == 'cifar10':
    #     data_loader = DataLoader(
    #         datasets.CIFAR10('data/cifar10', train=True, download=True, transform=transform),
    #         batch_size=batch_size, shuffle=True)
    # elif dataset == 'svhn':
    #     data_loader = DataLoader(
    #         datasets.SVHN('data/svhn', split=split, download=True, transform=transform),
    #         batch_size=batch_size, shuffle=True)
    # elif dataset == 'stl10':
    #     data_loader = DataLoader(
    #         datasets.STL10('data/stl10', split=split, download=True, transform=transform),
    #         batch_size=batch_size, shuffle=True)
    # elif dataset == 'lsun-bed':
    #     data_loader = DataLoader(
    #         datasets.LSUN('data/lsun', classes=['bedroom_train'], transform=transform),
    #         batch_size=batch_size, shuffle=True)


    return data_loader