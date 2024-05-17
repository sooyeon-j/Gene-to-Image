from contextlib import nullcontext
import os
import copy
import numpy as np
import torch
from torch import optim
import torch.nn as nn

from types import SimpleNamespace
from fastprogress import progress_bar, master_bar
from utils import *
from torchvision import models

from generative.networks.nets import AutoencoderKL, DiffusionModelUNet


def load_seed(seed):
    # Random Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed

def load_device():
    if torch.cuda.is_available():
        device = list(range(torch.cuda.device_count()))
    else:
        device = 'cpu'
    return device

def load_model_params(config, type= 'ae'):
    if type == 'ae':
        config = config.autoencoder
        params = {'spatial_dims': config.spatial_dims, 'in_channels': config.in_channels, 'out_channels': config.out_channels,
                    'num_channels': config.num_channels, 'latent_channels': config.latent_channels,
                    'num_res_blocks': config.num_res_blocks, 'attention_levels': config.attention_levels,
                    'with_encoder_nonlocal_attn': config.with_encoder_nonlocal_attn, 'with_decoder_nonlocal_attn': config.with_decoder_nonlocal_attn}
    elif type == 'diffusion':
        config = config.diffusion
        params = {'spatial_dims': config.spatial_dims, 'in_channels': config.in_channels, 'out_channels': config.out_channels,
                    'num_channels': config.num_channels, 'num_res_blocks': config.num_res_blocks, 
                    'attention_levels': config.attention_levels, 'num_head_channels': config.num_head_channels,
                    'with_conditioning': config.with_conditioning, 'num_context_embeds': config.num_context_embeds,
                    'num_class_embeds': config.num_class_embeds, 'cross_attention_dim': config.cross_attention_dim}
    return params

def load_model(model, device):
    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
        model = model.to(f'cuda:{device[0]}')
    return model

def load_batch(batch, device, type):
    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device
    img_b = batch[0].to(device_id)
    label_b = batch[1].to(device_id)
    if type == 'diffusion':
        snp_b = batch[2].to(device_id)
        return img_b, snp_b, label_b
    return img_b, label_b

def load_ckpt(config, device, ae=True):
    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device
    ckpt_dict = {}
    if ae:
        path = f'./checkpoints/{config.data.data}/{config.ae_ckpt}_ae.pth'
    else:
        path = f'./checkpoints/{config.data.data}/{config.run_name}_unet.pth'
    ckpt = torch.load(path, map_location=device_id)
    print(f'{path} loaded')
    ckpt_dict= {'params': ckpt['model_params'], 'state_dict': ckpt['state_dict']}

    return ckpt_dict

def load_model_from_ckpt(params, state_dict, device, ae=True):
    params_ = params.copy()
    if ae:
        model = AutoencoderKL(**params_)
    else:
        model = DiffusionModelUNet(**params_)
    if 'module.' in list(state_dict.keys())[0]:
        # strip 'module.' at front; for DataParallel models
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
        model = model.to(f'cuda:{device[0]}')
    return model

