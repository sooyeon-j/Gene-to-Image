import os
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import wandb
import logging
import time

from parsers.parser import Parser
from parsers.config import get_config
from PIL import Image
from monai import transforms
from monai.config import print_config
from monai.data import DataLoader, Dataset
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from contextlib import nullcontext
from itertools import combinations
from utils import *
from loaders import *
from sampling import Sampling

from generative.inferers import LatentDiffusionInferer
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler
from generative.metrics import FIDMetric, MMDMetric, MultiScaleSSIMMetric, SSIMMetric, PSNRMetric

# from baseline.classifier import Classifier
# from metric import Metric

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class Training(object):
    def __init__(self, config):
        super(Training, self).__init__()
        
        print_config()

        load_seed(config.seed)

        mk_folders(config)
        
        self.device = load_device()
        self.device_0 = f'cuda:{self.device[0]}' if isinstance(self.device, list) else self.device
        self.config = config
        self.ae_train_loader, self.ae_val_loader = get_mri(config, config.data.ae_dataset_path, type= 'ae')
        self.dm_train_loader, self.dm_val_loader, self.dm_test_loader = get_mri(config, config.data.dm_dataset_path, config.data.dm_snp_path, type= 'diffusion')
        
    def autoencoder(self):
        params = load_model_params(self.config, type= 'ae')
        autoencoderkl = AutoencoderKL(**params.copy())
        self.autoencoderkl = load_model(autoencoderkl, self.device)

        perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")
        perceptual_loss.to(self.device_0)
        perceptual_weight = 0.001

        discriminator = PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=3, out_channels=3)
        discriminator = discriminator.to(self.device_0)

        adv_loss = PatchAdversarialLoss(criterion="least_squares")
        adv_weight = 0.01

        optimizer_g = torch.optim.Adam(self.autoencoderkl.parameters(), lr=1e-4)
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=5e-4)

        scaler_g = torch.cuda.amp.GradScaler()
        scaler_d = torch.cuda.amp.GradScaler()

        kl_weight = 1e-6
        n_epochs = self.config.train.ae_epochs
        val_interval = self.config.train.ae_val_interval
        autoencoder_warm_up_n_epochs = self.config.train.autoencoder_warm_up_n_epochs

        epoch_recon_losses = []
        epoch_gen_losses = []
        epoch_disc_losses = []
        val_recon_losses = []
        intermediary_images = []
        num_example_images = 4

        for epoch in range(n_epochs):
            self.autoencoderkl.train()
            discriminator.train()
            epoch_loss = 0
            gen_epoch_loss = 0
            disc_epoch_loss = 0
            progress_bar = tqdm(enumerate(self.ae_train_loader), total=len(self.ae_train_loader), ncols=110)
            progress_bar.set_description(f"Epoch {epoch}")
            for step, batch in progress_bar:
                images, labels = load_batch(batch, self.device, type= 'ae') #(64, 3, 128, 128)
                optimizer_g.zero_grad(set_to_none=True)

                with autocast(enabled=True):
                    reconstruction, z_mu, z_sigma = self.autoencoderkl(images)
                    # (64, 3, 128, 128), (64, 3, 32, 32), (64, 3, 32, 32)
                    # (64, 3, 224, 192), (64, 3, 56, 48), (64, 3, 56, 48)
                    recons_loss = F.l1_loss(reconstruction.float(), images.float())
                    p_loss = perceptual_loss(reconstruction.float(), images.float())
                    kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
                    kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
                    loss_g = recons_loss + (kl_weight * kl_loss) + (perceptual_weight * p_loss)

                    if epoch > autoencoder_warm_up_n_epochs:
                        logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                        generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                        loss_g += adv_weight * generator_loss

                scaler_g.scale(loss_g).backward()
                scaler_g.step(optimizer_g)
                scaler_g.update()

                if epoch > autoencoder_warm_up_n_epochs:
                    with autocast(enabled=True):
                        optimizer_d.zero_grad(set_to_none=True)

                        logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                        loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                        logits_real = discriminator(images.contiguous().detach())[-1]
                        loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                        loss_d = adv_weight * discriminator_loss

                    scaler_d.scale(loss_d).backward()
                    scaler_d.step(optimizer_d)
                    scaler_d.update()

                epoch_loss += recons_loss.item()
                if epoch > autoencoder_warm_up_n_epochs:
                    gen_epoch_loss += generator_loss.item()
                    disc_epoch_loss += discriminator_loss.item()
                    wandb.log({
                        "gen_loss": gen_epoch_loss / (step + 1),
                        "disc_loss": disc_epoch_loss / (step + 1),
                    })

                progress_bar.set_postfix(
                    {
                        "recons_loss": epoch_loss / (step + 1),
                        "gen_loss": gen_epoch_loss / (step + 1),
                        "disc_loss": disc_epoch_loss / (step + 1),
                    }
                )
                wandb.log({
                        "recons_loss": epoch_loss / (step + 1),
                    })
                
            epoch_recon_losses.append(epoch_loss / (step + 1))
            epoch_gen_losses.append(gen_epoch_loss / (step + 1))
            epoch_disc_losses.append(disc_epoch_loss / (step + 1))

            if (epoch + 1) % val_interval == 0:
                self.autoencoderkl.eval()
                val_loss = 0
                with torch.no_grad():
                    for val_step, batch in enumerate(self.ae_val_loader, start=1):
                        images, labels = load_batch(batch, self.device, type= 'ae')

                        with autocast(enabled=True):
                            reconstruction, z_mu, z_sigma = self.autoencoderkl(images)
                            recons_loss = F.l1_loss(images.float(), reconstruction.float())

                        val_loss += recons_loss.item()

                val_loss /= val_step
                val_recon_losses.append(val_loss)
                print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")
                wandb.log({"AE_val_loss": val_loss})
        progress_bar.close()

        del discriminator
        del perceptual_loss
        torch.cuda.empty_cache()
        
        torch.save({
            'model_params': params,
            'state_dict': self.autoencoderkl.state_dict()
        }, f'./checkpoints/{self.config.data.data}/{self.config.run_name}_ae.pth')
        
        # Plot last 5 evaluations
        reconstructions = torch.reshape(reconstruction[:num_example_images, 0], (self.config.data.height * num_example_images, self.config.data.width)).T # (512, 128) -> (128, 512)
        reconstruction_ = reconstructions.clamp(0, 1)

        plt.figure(figsize=(12, 3))
        path = os.path.join("results", self.config.run_name, "AE.jpg")
        plt.imsave(path, reconstruction_.detach().cpu().numpy())


    def diffusion(self):
        
        def subtract_mean(x: torch.Tensor) -> torch.Tensor:
            mean = [0.406, 0.456, 0.485]
            x[:, 0, :, :] -= mean[0]
            x[:, 1, :, :] -= mean[1]
            x[:, 2, :, :] -= mean[2]
            return x
        
        def spatial_average(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
            return x.mean([2, 3], keepdim=keepdim)

        def get_features(image):
            # If input has just 1 channel, repeat channel to have 3 channels
            if image.shape[1]:
                image = image.repeat(1, 3, 1, 1)

            # Change order from 'RGB' to 'BGR'
            image = image[:, [2, 1, 0], ...]

            # Subtract mean used during training
            image = subtract_mean(image)

            # Get model outputs
            with torch.no_grad():
                feature_image = radnet.forward(image)
                # flattens the image spatially
                feature_image = spatial_average(feature_image, keepdim=False)

            return feature_image
                
        if self.config.stage == 'diffusion':
            ckpt_dict_ae = load_ckpt(self.config, self.device, ae=True)
            ae = load_model_from_ckpt(ckpt_dict_ae['params'], ckpt_dict_ae['state_dict'], self.device, ae=True)
            self.autoencoderkl = ae
            
        params = load_model_params(self.config, type= 'diffusion')

        unet = DiffusionModelUNet(**params.copy())
        scheduler = DDPMScheduler(num_train_timesteps=self.config.train.timesteps, schedule="linear_beta", beta_start= self.config.train.beta_start, beta_end= self.config.train.beta_end)

        # ### Scaling factor
        #
        # As mentioned in Rombach et al. [1] Section 4.3.2 and D.1, the signal-to-noise ratio (induced by the scale of the latent space) can affect the results obtained with the LDM, if the standard deviation of the latent space distribution drifts too much from that of a Gaussian. For this reason, it is best practice to use a scaling factor to adapt this standard deviation.
        #
        # _Note: In case where the latent space is close to a Gaussian distribution, the scaling factor will be close to one, and the results will not differ from those obtained when it is not used._
        #
        
        with torch.no_grad():
            with autocast(enabled=True):
                # z = self.autoencoderkl.encode_stage_2_inputs(first(self.dm_train_loader)["image"].to(self.device_0))
                z = self.autoencoderkl(first(self.dm_train_loader)[0].to(self.device_0), 'encode_2_inputs')

        print(f"Scaling factor set to {1/torch.std(z)}")
        scale_factor = 1 / torch.std(z)

        inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

        optimizer = torch.optim.AdamW(unet.parameters(), lr= self.config.train.lr, weight_decay=self.config.train.weight_decay)
        
        unet = load_model(unet, self.device)

        radnet = torch.hub.load("Warvito/radimagenet-models", model="radimagenet_resnet50", verbose=True)
        radnet.to(self.device_0)
        radnet.eval()
                
        fid = FIDMetric()
        ms_ssim = MultiScaleSSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=4)
        ssim = SSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=4)
        psnr = PSNRMetric(max_val= 1.0)
        
        n_epochs = self.config.train.diff_epochs
        val_interval = self.config.train.diff_val_interval
        epoch_losses = []
        val_losses = []
        scaler = GradScaler()
        path = os.path.join("results", self.config.run_name, "diffusion")

        for epoch in range(n_epochs):
            unet.train()
            self.autoencoderkl.eval()
            epoch_loss = 0
            progress_bar = tqdm(enumerate(self.dm_train_loader), total=len(self.dm_train_loader), ncols=70)
            progress_bar.set_description(f"Epoch {epoch}")
            for step, batch in progress_bar:
                images, snps, labels = load_batch(batch, self.device, type= 'diffusion')
                # labels = labels.unsqueeze(-1).unsqueeze(-1).type(torch.float32)
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=True):
                    # z_mu, z_sigma = self.autoencoderkl(images, stage= 'encode') # (64, 3, 32, 32), (64, 3, 32, 32)
                    z = self.autoencoderkl(images, 'encode_2_inputs')
                    noise = torch.randn_like(z).to(self.device_0)
                    timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()
                    noise_pred = inferer(
                        inputs=images, diffusion_model=unet, noise=noise, timesteps=timesteps, 
                        autoencoder_model=self.autoencoderkl,
                        # condition= labels,
                        labels= labels
                        )
                    loss = F.mse_loss(noise_pred.float(), noise.float())

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()

                progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
                wandb.log({"diffusion_loss": epoch_loss / (step + 1)})
            epoch_losses.append(epoch_loss / (step + 1))
            
            if (epoch + 1) % val_interval == 0:
                unet.eval()
                val_loss = 0
                with torch.no_grad():
                    for val_step, batch in enumerate(self.dm_val_loader, start=1):
                        images, snps, labels = load_batch(batch, self.device, type= 'diffusion')
                        # labels = labels.unsqueeze(-1).unsqueeze(-1).type(torch.float32)
                        with autocast(enabled=True):
                            # z_mu, z_sigma = self.autoencoderkl(images, stage='encode')
                            z = self.autoencoderkl(images, 'encode_2_inputs')
                            noise = torch.randn_like(z).to(self.device_0)
                            timesteps = torch.randint(
                                0, inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device
                            ).long()
                            noise_pred = inferer(
                                inputs=images,
                                diffusion_model=unet,
                                noise=noise,
                                timesteps=timesteps,
                                autoencoder_model= self.autoencoderkl,
                                # condition= labels,
                                labels= labels
                            )

                            loss = F.mse_loss(noise_pred.float(), noise.float())

                        val_loss += loss.item()
                val_loss /= val_step
                val_losses.append(val_loss)
                print(f"Epoch {epoch} val loss: {val_loss:.4f}")
                wandb.log({"Unet_val_loss": val_loss})

                synth_features = []
                real_features = []

                ms_ssim_scores = []
                ssim_scores = []
                psnr_scores = []

                # Sampling image during training
                batch = next(iter(self.dm_val_loader))
                images, snps, labels = load_batch(batch, self.device, type= 'diffusion')
                # labels = labels.unsqueeze(-1).unsqueeze(-1).type(torch.float32)
                n = len(images)
                z = torch.randn((n, 3, 64, 64))
                z = z.to(self.device_0)
                scheduler.set_timesteps(num_inference_steps=self.config.sample.sample_steps)
                with autocast(enabled=True):
                    decoded = inferer.sample(
                        input_noise=z, diffusion_model=unet, scheduler=scheduler, autoencoder_model= self.autoencoderkl, 
                        # conditioning= labels,
                        labels= labels,
                        guidance_scale= self.config.sample.guidance_scale
                    ) # (1, 3, 64, 64)
                    
                    # images = (images.clamp(-1, 1) + 1) / 2
                    # decoded = (decoded.clamp(-1, 1) + 1) / 2
                    psnr_scores.append(psnr(decoded, images))
                    
                    real_eval_feats = get_features(images)
                    real_features.append(real_eval_feats)
                    
                    synth_eval_feats = get_features(decoded)
                    synth_features.append(synth_eval_feats)
                    
                    idx_pairs = list(combinations(range(n), 2))
                    for idx_a, idx_b in idx_pairs:
                        ms_ssim_scores.append(ms_ssim(decoded[[idx_a]], decoded[[idx_b]]))
                        ssim_scores.append(ssim(decoded[[idx_a]], decoded[[idx_b]]))
                
                psnr_scores = torch.cat(psnr_scores, dim=0)
                wandb.log({"PSNR Scores": psnr_scores.mean()})
                                
                synth_features = torch.vstack(synth_features)
                real_features = torch.vstack(real_features)

                fid_res = fid(synth_features, real_features)
                wandb.log({"FID Scores": fid_res.item()})
                
                ms_ssim_scores = torch.cat(ms_ssim_scores, dim=0)
                ssim_scores = torch.cat(ssim_scores, dim=0)
                wandb.log({"MS-SSIM Metric": ms_ssim_scores.mean(), "SSIM": ssim_scores.mean()})
                
                decoded = decoded.clamp(0, 1)

                save_images(decoded[0], os.path.join(path, f"intermediate_{epoch}_{labels[0].item()}.jpg"))
                
                # save_images(decoded, os.path.join(path, f"{epoch}.jpg"))
                wandb.log({"intermediate_images": [wandb.Image(decoded[0].permute(1,2,0).squeeze().cpu().numpy())]})
                
                ckpt_path = os.path.join("checkpoints", self.config.data.data, self.config.run_name)
                if not os.path.exists(ckpt_path):
                    os.makedirs(ckpt_path)
                torch.save({
                    'model_params': params,
                    'state_dict': unet.state_dict()
                }, f'./checkpoints/{self.config.data.data}/{self.config.run_name}/{epoch}_unet.pth')
                        
        progress_bar.close()
        
        torch.save({
            'model_params': params,
            'state_dict': unet.state_dict()
        }, f'./checkpoints/{self.config.data.data}/{self.config.run_name}_unet.pth')
                
        mk_folders(self.config, train= False)                        
        unet.eval()
        for step, batch in list(enumerate(self.dm_test_loader)):
            logging.info(f"Epoch: {step}, Sampling a new train_images....")
            with torch.no_grad():
                images, snps, labels = load_batch(batch, self.device, type= 'diffusion')
                # labels = labels.unsqueeze(-1).unsqueeze(-1).type(torch.float32)
                n = len(images)
                z = torch.randn((n, 3, 64, 64))
                z = z.to(self.device_0)
                scheduler.set_timesteps(num_inference_steps=self.config.sample.sample_steps)
                with autocast(enabled=True):
                    decoded = inferer.sample(
                        input_noise=z, diffusion_model=unet, autoencoder_model=self.autoencoderkl,
                        scheduler=scheduler,
                        # conditioning= labels,
                        labels= labels,
                        guidance_scale= self.config.sample.guidance_scale
                        )
                    
            decoded = decoded.clamp(0, 1)

            path = os.path.join("results", self.config.run_name, "samples", self.config.ts)  
                     
            for i in range(n):
                if labels[0].item() == 0:
                    save_images(decoded[i], os.path.join(path, "AD", f"{step}_{i}.jpg"))
                elif labels[0].item() == 1:
                    save_images(decoded[i], os.path.join(path, "CN", f"{step}_{i}.jpg"))
                elif labels[0].item() == 2:
                    save_images(decoded[i], os.path.join(path, "MCI", f"{step}_{i}.jpg"))
            
            # for i in range(n):
            #     save_images(decoded[i], os.path.join(path, f"sample_{i}_{labels[i].item()}.jpg"))
            # decoded = decoded.clamp(0, 1)
            #     # save_images(decoded, os.path.join(path, f"{epoch}.jpg"))
            
            wandb.log({"sampled_images": [wandb.Image(decoded[i].permute(1,2,0).cpu().numpy()) for i in range(n)]})

if __name__ == '__main__':
    
    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    args = Parser().parse()
    config = get_config(args)
    config.ts = ts
    
    if args.type == 'train':
        training = Training(config)
        with wandb.init(project= "MRI_ldm", group= "LABEL",  config= config) if config.train.use_wandb else nullcontext():
            if args.stage == 'both':
                training.autoencoder()
                training.diffusion()
            elif args.stage == 'diffusion':
                training.diffusion()
            
    elif args.type == 'sample':
        mk_folders(config, train= False)
        sampling = Sampling(config)
        sampling.sample()
        
    # elif args.type == 'classify':
    #     classification = Classifier(config)
    #     classification.classify()
        
    elif args.type == 'metric':
        metric = Metric(config)
        metric.metric()