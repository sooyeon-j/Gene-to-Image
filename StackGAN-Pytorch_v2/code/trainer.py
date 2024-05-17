from __future__ import print_function
from six.moves import range
from PIL import Image

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.optim as optim
import os
import time

import numpy as np
import torchfile
import wandb
import pytorch_fid_wrapper as pfw

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import weights_init
from miscc.utils import save_img_results, save_model
from miscc.utils import KL_loss
from miscc.utils import compute_discriminator_loss, compute_generator_loss

from torch.utils.tensorboard import summary
from torch.utils.tensorboard import FileWriter

from generative.metrics import FIDMetric, MMDMetric, MultiScaleSSIMMetric, SSIMMetric, PSNRMetric
from itertools import combinations

class GANTrainer(object):
    def __init__(self, output_dir):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            self.summary_writer = FileWriter(self.log_dir)

        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True

    # ############# For training stageI GAN #############
    def load_network_stageI(self):
        from model import STAGE1_G, STAGE1_D
        netG = STAGE1_G()
        netG.apply(weights_init)
        # print(netG)
        netD = STAGE1_D()
        netD.apply(weights_init)
        # print(netD)

        if cfg.NET_G != '':
            state_dict = \
                torch.load(cfg.NET_G,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_G)
        if cfg.NET_D != '':
            state_dict = \
                torch.load(cfg.NET_D,
                           map_location=lambda storage, loc: storage)
            netD.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_D)
        if cfg.CUDA:
            netG.cuda()
            netD.cuda()
        return netG, netD

    # ############# For training stageII GAN  #############
    def load_network_stageII(self):
        from model import STAGE1_G, STAGE2_G, STAGE2_D

        Stage1_G = STAGE1_G()
        netG = STAGE2_G(Stage1_G)
        netG.apply(weights_init)
        # print(netG)
        if cfg.NET_G != '':
            state_dict = \
                torch.load(cfg.NET_G,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_G)
        elif cfg.STAGE1_G != '':
            state_dict = \
                torch.load(cfg.STAGE1_G,
                           map_location=lambda storage, loc: storage)
            netG.STAGE1_G.load_state_dict(state_dict)
            print('Load from: ', cfg.STAGE1_G)
        else:
            print("Please give the Stage1_G path")
            return

        netD = STAGE2_D()
        netD.apply(weights_init)
        if cfg.NET_D != '':
            state_dict = \
                torch.load(cfg.NET_D,
                           map_location=lambda storage, loc: storage)
            netD.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_D)
        # print(netD)

        if cfg.CUDA:
            netG.cuda()
            netD.cuda()
        return netG, netD

    def train(self, data_loader, stage=1):
        if stage == 1:
            netG, netD = self.load_network_stageI()
        else:
            netG, netD = self.load_network_stageII()

        nz = cfg.Z_DIM

        generator_lr = cfg.TRAIN.GENERATOR_LR
        discriminator_lr = cfg.TRAIN.DISCRIMINATOR_LR
        lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH
        optimizerD = \
            optim.Adam(netD.parameters(),
                       lr=cfg.TRAIN.DISCRIMINATOR_LR, betas=(cfg.TRAIN.DISCRIMINATOR_COEFF.beta1, cfg.TRAIN.DISCRIMINATOR_COEFF.beta2))
        netG_para = []
        for p in netG.parameters():
            if p.requires_grad:
                netG_para.append(p)
        optimizerG = optim.Adam(netG_para,
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(cfg.TRAIN.GENERATOR_COEFF.beta1, cfg.TRAIN.GENERATOR_COEFF.beta2))
        count = 0
        for epoch in range(self.max_epoch):
            start_t = time.time()
            if epoch % lr_decay_step == 0 and epoch > 0:
                generator_lr *= 0.5
                for param_group in optimizerG.param_groups:
                    param_group['lr'] = generator_lr
                discriminator_lr *= 0.5
                for param_group in optimizerD.param_groups:
                    param_group['lr'] = discriminator_lr

            for i, data in enumerate(data_loader, 0):
                ######################################################
                # (1) Prepare training data
                ######################################################
                real_img_cpu, _, embedding = data # img, label, snp
                real_imgs = Variable(real_img_cpu)
                embedding = Variable(embedding)
                
                bs = real_imgs.shape[0]
                noise = Variable(torch.FloatTensor(bs, nz))
                with torch.no_grad():
                    fixed_noise = \
                        Variable(torch.FloatTensor(bs, nz).normal_(0, 1))
                real_labels = Variable(torch.FloatTensor(bs).fill_(1))
                fake_labels = Variable(torch.FloatTensor(bs).fill_(0))
                if cfg.CUDA:
                    real_imgs = real_imgs.cuda()
                    embedding = embedding.cuda()
                    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
                    real_labels, fake_labels = real_labels.cuda(), fake_labels.cuda()
                
                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                inputs = (embedding, noise)
                _, fake_imgs, mu, logvar = \
                    nn.parallel.data_parallel(netG, inputs, self.gpus)

                ############################
                # (3) Update D network
                ###########################
                netD.zero_grad()
                errD, errD_real, errD_wrong, errD_fake = \
                    compute_discriminator_loss(netD, real_imgs, fake_imgs,
                                            real_labels, fake_labels,
                                            mu, self.gpus)
                errD.backward()
                optimizerD.step()
                
                ############################
                # (2) Update G network
                ###########################
                netG.zero_grad()
                errG = compute_generator_loss(netD, fake_imgs,
                                                real_labels, mu, self.gpus)
                kl_loss = KL_loss(mu, logvar)
                errG_total = errG + kl_loss * cfg.TRAIN.COEFF.KL
                errG_total.backward()
                optimizerG.step()

                # count = count + 1
                
            if epoch % self.snapshot_interval == 0:
                # summary_D = summary.scalar('D_loss', errD.item())
                # summary_D_r = summary.scalar('D_loss_real', errD_real)
                # summary_D_w = summary.scalar('D_loss_wrong', errD_wrong)
                # summary_D_f = summary.scalar('D_loss_fake', errD_fake)
                # summary_G = summary.scalar('G_loss', errG.item())
                # summary_KL = summary.scalar('KL_loss', kl_loss.item())

                # self.summary_writer.add_summary(summary_D, count)
                # self.summary_writer.add_summary(summary_D_r, count)
                # self.summary_writer.add_summary(summary_D_w, count)
                # self.summary_writer.add_summary(summary_D_f, count)
                # self.summary_writer.add_summary(summary_G, count)
                # self.summary_writer.add_summary(summary_KL, count)
                wandb.log({"D_loss": errD.item(), 
                            "D_loss_real": errD_real, 
                            "D_loss_wrong": errD_wrong, 
                            "D_loss_fake": errD_fake,
                            "G_loss": errG.item(),
                            "KL_loss": kl_loss.item()})

                # save the image result for each epoch
                with torch.no_grad():
                    inputs = (embedding, fixed_noise)
                    lr_fake, fake, _, _ = \
                        nn.parallel.data_parallel(netG, inputs, self.gpus)
                    save_img_results(real_img_cpu, fake, epoch, self.image_dir)
                    wandb.log({"intermediate_img": [wandb.Image(fake[0].permute(1,2,0).squeeze().cpu().numpy())] })
                    if lr_fake is not None:
                        save_img_results(None, lr_fake, epoch, self.image_dir)
                        
            end_t = time.time()
            print('''[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_KL: %.4f
                     Loss_real: %.4f Loss_wrong:%.4f Loss_fake %.4f
                     Total Time: %.2fsec
                  '''
                  % (epoch, self.max_epoch, i, len(data_loader),
                     errD.item(), errG.item(), kl_loss.item(),
                     errD_real, errD_wrong, errD_fake, (end_t - start_t)))
            if epoch % self.snapshot_interval == 0:
                save_model(netG, netD, epoch, self.model_dir)
        #
        save_model(netG, netD, self.max_epoch, self.model_dir)
        #
        self.summary_writer.close()
        
    def get_normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = [0.406, 0.456, 0.485]
        std = [0.225, 0.224, 0.229]
        # x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        # x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        # x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]
        
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) 
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1])
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2])
        
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
        image = self.get_normalize(image)
        
        # Get model outputs
        with torch.no_grad():
            feature_image = self.radnet.forward(image)
            # flattens the image spatially
            feature_image = self.spatial_average(feature_image, keepdim=False)

        return feature_image
        
    def sample(self, dataloader, stage=1):
        
        self.radnet = torch.hub.load("Warvito/radimagenet-models", model="radimagenet_resnet50", verbose=True)
        self.radnet.to('cuda')
        self.radnet.eval()
                
        self.fid = FIDMetric()
        self.ms_ssim = MultiScaleSSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=4)
        self.ssim = SSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=4)
        self.psnr = PSNRMetric(max_val= 1.0)
        
        if stage == 1:
            netG, _ = self.load_network_stageI()
        else:
            netG, _ = self.load_network_stageII()
        netG.eval()
        
        # path to save generated samples
        save_dir = os.path.join(cfg.NET_G[:cfg.NET_G.find('Model')-1], "Sample", cfg.ts) 
        mkdir_p(save_dir, sampling=True)
        
        nz = cfg.Z_DIM
        
        fid_radnet = []
        fid_imgnet = []
        ms_ssim_total = []
        ssim_total = []
        psnr_total = []
        
        for epoch in range(cfg.SAMPLE.EPOCH):     
            synth_features = []
            real_features = []
            fid_scores = []
            ms_ssim_scores = []
            ssim_scores = []
            psnr_scores = []            

            for i, data in enumerate(dataloader, 0):
                with torch.inference_mode():
                    ######################################################
                    # (1) Prepare training data
                    ######################################################
                    real_img_cpu, labels, embedding = data # img, label, snp

                    real_imgs = Variable(real_img_cpu)
                    embedding = Variable(embedding)
                    
                    bs = real_imgs.shape[0]
                    noise = Variable(torch.FloatTensor(bs, nz))
                    
                    if cfg.CUDA:
                        noise = noise.cuda()
                        embedding = embedding.cuda()
                        real_imgs = real_imgs.cuda()

                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.data.normal_(0, 1)
                    inputs = (embedding, noise)
                    _, fake_imgs, mu, logvar = \
                        nn.parallel.data_parallel(netG, inputs, self.gpus)
                    
                    real_imgs = (real_imgs + 1.0) / 2.0
                    fake_imgs = (fake_imgs + 1.0) / 2.0
                    # fake_imgs = fake_imgs.clamp(0, 1)
                    
                    # real_imgs = real_imgs.type(torch.float32)
                    # fake_imgs = fake_imgs.type(torch.float32)

                    psnr_scores.append(self.psnr(fake_imgs, real_imgs))

                    real_eval_feats = self.get_features(real_imgs)
                    real_features.append(real_eval_feats)
                    
                    synth_eval_feats = self.get_features(fake_imgs)
                    synth_features.append(synth_eval_feats)
                    
                    pfw.set_config(batch_size=bs, device='cuda')
                    fid_score = pfw.fid(fake_imgs, real_imgs)
                    fid_scores.append(fid_score)          
                                  
                    idx_pairs = list(combinations(range(bs), 2))
                    for idx_a, idx_b in idx_pairs:
                        ms_ssim_scores.append(self.ms_ssim(fake_imgs[[idx_a]], fake_imgs[[idx_b]]))
                        ssim_scores.append(self.ssim(fake_imgs[[idx_a]], fake_imgs[[idx_b]]))
                    
                    wandb.log({"sampled_images": [wandb.Image(fake_imgs[0].permute(1,2,0).cpu().numpy())]})
                    
                    for j in range(bs):
                        if labels[j].item() == 0:
                            vutils.save_image(
                                fake_imgs[j], os.path.join(save_dir, "AD", f"fake_{epoch}_{i}_{j}.png"),
                                normalize=True)
                            # vutils.save_image(
                            #     real_img_cpu[j], os.path.join(save_dir, "AD", f"real_{epoch}_{i}_{j}.png"),
                            #     normalize=True)
                        elif labels[j].item() == 1:
                            vutils.save_image(
                                fake_imgs[j], os.path.join(save_dir, "CN", f"fake_{epoch}_{i}_{j}.png"),
                                normalize=True)
                            # vutils.save_image(
                            #     real_img_cpu[j], os.path.join(save_dir, "CN", f"real_{epoch}_{i}_{j}.png"),
                            #     normalize=True)    
                        elif labels[j].item() == 2:
                            vutils.save_image(
                                fake_imgs[j], os.path.join(save_dir, "MCI", f"fake_{epoch}_{i}_{j}.png"),
                                normalize=True)
                            # vutils.save_image(
                            #     real_img_cpu[j], os.path.join(save_dir, "MCI", f"real_{epoch}_{i}_{j}.png"),
                            #     normalize=True)        
                                             
                        # save_name = '%s/%d.png' % (save_dir, epoch + i)
                        # im = fake_imgs[i].data.cpu().numpy()
                        # im = im * 255.0
                        # im = im.astype(np.uint8)
                        # # print('im', im.shape)
                        # im = np.transpose(im, (1, 2, 0))
                        # # print('im', im.shape)
                        # im = Image.fromarray(im)
                        # im.save(save_name)

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
         
        # Load text embeddings generated from the encoder
        # t_file = torchfile.load(datapath)
        # captions_list = t_file.raw_txt
        # embeddings = np.concatenate(t_file.fea_txt, axis=0)
        # num_embeddings = len(captions_list)
        # print('Successfully load sentences from: ', datapath)
        # print('Total number of sentences:', num_embeddings)
        # print('num_embeddings:', num_embeddings, embeddings.shape)
