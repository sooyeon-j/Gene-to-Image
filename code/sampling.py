from contextlib import nullcontext
import os
import numpy as np
import torch

from fastprogress import progress_bar, master_bar
from utils import *
from loaders import *
from generative.networks.schedulers import DDPMScheduler
from generative.inferers import LatentDiffusionInferer
from generative.metrics import FIDMetric, MMDMetric, MultiScaleSSIMMetric, SSIMMetric, PSNRMetric
from torch.cuda.amp import autocast
from itertools import combinations

import logging
import wandb
import pytorch_fid_wrapper as pfw

class Sampling(object):
    def __init__(self, config):
        super(Sampling, self).__init__()
        
        load_seed(config.seed)
        
        self.config = config
        self.device = load_device()
    
        self.beta_start = self.config.sample.beta_start
        self.beta_end = self.config.sample.beta_end
        self.sample_steps = self.config.sample.sample_steps
        
        self.dataloader = get_mri(self.config, self.config.data.dm_dataset_path, self.config.data.dm_snp_path, type= 'sample')
        
        self.radnet = torch.hub.load("Warvito/radimagenet-models", model="radimagenet_resnet50", verbose=True)
        self.radnet.to('cuda')
        self.radnet.eval()
                
        self.fid = FIDMetric()
        self.ms_ssim = MultiScaleSSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=4)
        self.ssim = SSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=4)
        self.psnr = PSNRMetric(max_val= 1.0)
        
    def subtract_mean(self, x: torch.Tensor) -> torch.Tensor:
        mean = [0.406, 0.456, 0.485]
        x[:, 0, :, :] -= mean[0]
        x[:, 1, :, :] -= mean[1]
        x[:, 2, :, :] -= mean[2]
        return x
    
    def spatial_average(self, x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
        return x.mean([2, 3], keepdim=keepdim)

    def get_features(self, image):
        # If input has just 1 channel, repeat channel to have 3 channels
        if image.shape[1]:
            image = image.repeat(1, 3, 1, 1)

        # Change order from 'RGB' to 'BGR'
        image = image[:, [2, 1, 0], ...]

        # Subtract mean used during training
        image = self.subtract_mean(image)

        # Get model outputs
        with torch.no_grad():
            feature_image = self.radnet.forward(image)
            # flattens the image spatially
            feature_image = self.spatial_average(feature_image, keepdim=False)

        return feature_image
    
    def sample(self, device= 'cuda'):    
        scheduler = DDPMScheduler(num_train_timesteps=self.config.sample.sample_steps, schedule="linear_beta", beta_start= self.config.sample.beta_start, beta_end= self.config.sample.beta_end)
        inferer = LatentDiffusionInferer(scheduler)
        self.ckpt_dict_ae = load_ckpt(self.config, self.device, ae=True)
        self.ckpt_dict_unet = load_ckpt(self.config, self.device, ae=False)
        self.ae = load_model_from_ckpt(self.ckpt_dict_ae['params'], self.ckpt_dict_ae['state_dict'], self.device, ae=True)
        self.unet = load_model_from_ckpt(self.ckpt_dict_unet['params'], self.ckpt_dict_unet['state_dict'], self.device, ae=False)
        epochs = self.config.sample.epochs
        
        with wandb.init(project="MRI_ldm", group="sampling", config=self.config) if self.config.sample.use_wandb else nullcontext():
            self.unet.eval()
            path = os.path.join("results", self.config.run_name, "samples", self.config.ts)
            fid_radnet = []
            fid_imgnet = []
            ms_ssim_total = []
            ssim_total = []
            psnr_total = []
            
            for epoch in range(epochs):
                synth_features = []
                real_features = []
                fid_scores = []
                ms_ssim_scores = []
                ssim_scores = []
                psnr_scores = []
                
                for step, batch in list(enumerate(self.dataloader)):
                    logging.info(f"Epoch: {epoch}, Sampling a new train_images....")
                    with torch.inference_mode():
                        images, snps, labels = load_batch(batch, self.device, type= 'diffusion')
                        # labels = labels.unsqueeze(-1).unsqueeze(-1).type(torch.float32)
                        n = len(images)
                        z = torch.randn((n, 3, 64, 64)).to(device)
                        scheduler.set_timesteps(num_inference_steps=self.config.sample.sample_steps)
                        
                        if self.config.diffusion.mode == 'concat':
                            snps = snps.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, z.shape[-2], z.shape[-1])
                
                        with autocast(enabled=True):
                            decoded = inferer.sample(
                                input_noise=z, diffusion_model=self.unet, autoencoder_model=self.ae,
                                scheduler=scheduler, 
                                conditioning=snps,
                                # conditioning=labels,
                                labels= labels,
                                guidance_scale= self.config.sample.guidance_scale,
                                mode = self.config.diffusion.mode
                                )

                        # images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
                        # decoded = torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0)
                        decoded = decoded.clamp(0, 1)
                        
                        decoded = decoded.type(torch.float32)

                        psnr_scores.append(self.psnr(decoded, images))
                        
                        real_eval_feats = self.get_features(images)
                        real_features.append(real_eval_feats)
                        
                        synth_eval_feats = self.get_features(decoded)
                        synth_features.append(synth_eval_feats)
                        
                        pfw.set_config(batch_size=n, device='cuda')
                        fid_score = pfw.fid(decoded, images)
                        fid_scores.append(fid_score)      
                                          
                        idx_pairs = list(combinations(range(n), 2))
                        for idx_a, idx_b in idx_pairs:
                            ms_ssim_scores.append(self.ms_ssim(decoded[[idx_a]], decoded[[idx_b]]))
                            ssim_scores.append(self.ssim(decoded[[idx_a]], decoded[[idx_b]]))
                            
                        # decoded = (decoded * 255).type(torch.uint8)

                        for i in range(images.shape[0]):
                            if labels[i].item() == 0:
                                save_images(decoded[i], os.path.join(path, "AD", f"{epoch}_{step}_{i}.png"))
                            elif labels[i].item() == 1:
                                save_images(decoded[i], os.path.join(path, "CN", f"{epoch}_{step}_{i}.png"))
                            elif labels[i].item() == 2:
                                save_images(decoded[i], os.path.join(path, "MCI", f"{epoch}_{step}_{i}.png"))

                            wandb.log({"sampled_images": [wandb.Image(decoded[i].permute(1,2,0).cpu().numpy())]})
                
                
                psnr_scores = torch.cat(psnr_scores, dim=0)
                wandb.log({"PSNR Scores": psnr_scores.mean()})
                
                synth_features = torch.vstack(synth_features)
                real_features = torch.vstack(real_features)
                
                fid_res = self.fid(synth_features, real_features)
                
                wandb.log({"FID Scores": fid_res.item()})
                
                fid_scores = np.array(fid_scores)
                wandb.log({"FID Scores_ori": fid_scores.mean()})         
                       
                ms_ssim_scores = torch.cat(ms_ssim_scores, dim=0)
                ssim_scores = torch.cat(ssim_scores, dim=0)
                wandb.log({"MS-SSIM Metric": ms_ssim_scores.mean(), "SSIM": ssim_scores.mean()})

                psnr_total.append(psnr_scores.mean().cpu())
                fid_radnet.append(fid_res.item())
                fid_imgnet.append(fid_scores.mean())
                ms_ssim_total.append(ms_ssim_scores.mean().cpu())
                ssim_total.append(ssim_scores.mean().cpu())
            
            print(f'PSNR Scores :{np.mean(psnr_total):.3f}')
            print(f'FID by imagenet Scores :{np.mean(fid_imgnet):.3f}')
            print(f'FID by radimagenet Scores :{np.mean(fid_radnet):.3f}')
            print(f'MS-SSIM Metric :{np.mean(ms_ssim_total):.3f}')
            print(f'SSIM Metric :{np.mean(ssim_total):.3f}') 