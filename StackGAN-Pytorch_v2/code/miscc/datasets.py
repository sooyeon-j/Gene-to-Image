from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch.utils.data as data
from PIL import Image
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import pandas as pd

from miscc.config import cfg

import torch
import os
import torchvision
import numpy as np

from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data
import pandas as pd
import torchvision.transforms as T

def dataloader(datapath, input_size, batch_size, split='train'):
    data_transforms = [T.Resize((input_size, input_size)), T.RandomHorizontalFlip(), T.ToTensor(),
                    #    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                       T.Lambda(lambda t: (t * 2) - 1)]
    data_transform = T.Compose(data_transforms)

    image_folder = torchvision.datasets.ImageFolder(root = datapath, transform = data_transform)
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
        label_tensor = torch.stack(label_list)
        snp_np = np.array(snp_list)
        snp_tensor = torch.tensor(snp_np, dtype = torch.float32)
        ds.append(TensorDataset(img_tensor, label_tensor, snp_tensor))

    # train_ds = TensorDataset(img_tensor[train_indices], label_tensor[train_indices], snp_tensor[train_indices])
    # val_ds = TensorDataset(img_tensor[val_indices], label_tensor[val_indices], snp_tensor[val_indices])
    # test_ds = TensorDataset(img_tensor[test_indices], label_tensor[test_indices], snp_tensor[test_indices])

    train_loader = DataLoader(ds[0], num_workers= 4, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(ds[1], num_workers= 4, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(ds[2], num_workers= 4, batch_size = batch_size, shuffle = False)

    return train_loader, val_loader, test_loader

class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train', embedding_type='cnn-rnn',
                 imsize=64, transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform
        self.imsize = imsize
        self.data = []
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)

        self.filenames = self.load_filenames(split_dir)
        self.embeddings = self.load_embedding(split_dir, embedding_type)
        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        # self.captions = self.load_all_captions()

    def get_img(self, img_path, bbox):
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        if bbox is not None:
            R = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - R)
            y2 = np.minimum(height, center_y + R)
            x1 = np.maximum(0, center_x - R)
            x2 = np.minimum(width, center_x + R)
            img = img.crop([x1, y1, x2, y2])
        load_size = int(self.imsize * 76 / 64)
        img = img.resize((load_size, load_size), PIL.Image.BILINEAR)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in xrange(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_all_captions(self):
        caption_dict = {}
        for key in self.filenames:
            caption_name = '%s/text/%s.txt' % (self.data_dir, key)
            captions = self.load_captions(caption_name)
            caption_dict[key] = captions
        return caption_dict

    def load_captions(self, caption_name):
        cap_path = caption_name
        with open(cap_path, "r") as f:
            captions = f.read().decode('utf8').split('\n')
        captions = [cap.replace("\ufffd\ufffd", " ")
                    for cap in captions if len(cap) > 0]
        return captions

    def load_embedding(self, data_dir, embedding_type):
        if embedding_type == 'cnn-rnn':
            embedding_filename = '/char-CNN-RNN-embeddings.pickle'
        elif embedding_type == 'cnn-gru':
            embedding_filename = '/char-CNN-GRU-embeddings.pickle'
        elif embedding_type == 'skip-thought':
            embedding_filename = '/skip-thought-embeddings.pickle'

        with open(data_dir + embedding_filename, 'rb') as f:
            embeddings = pickle.load(f)
            embeddings = np.array(embeddings)
            # embedding_shape = [embeddings.shape[-1]]
            print('embeddings: ', embeddings.shape)
        return embeddings

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f)
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames

    def __getitem__(self, index):
        key = self.filenames[index]
        # cls_id = self.class_id[index]
        #
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir

        # captions = self.captions[key]
        embeddings = self.embeddings[index, :, :]
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        img = self.get_img(img_name, bbox)

        embedding_ix = random.randint(0, embeddings.shape[0]-1)
        embedding = embeddings[embedding_ix, :]
        if self.target_transform is not None:
            embedding = self.target_transform(embedding)
        return img, embedding

    def __len__(self):
        return len(self.filenames)

