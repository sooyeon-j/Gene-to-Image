import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataloader import dataloader
from itertools import combinations
from metrics import FIDMetric, MMDMetric, MultiScaleSSIMMetric, SSIMMetric, PSNRMetric

import wandb
import pytorch_fid_wrapper as pfw

class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32, class_num=10):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim + self.class_num, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size[0] // 4) * (self.input_size[1] // 4)),
            nn.BatchNorm1d(128 * (self.input_size[0] // 4) * (self.input_size[1] // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        utils.initialize_weights(self)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_size[0] // 4), (self.input_size[1] // 4))
        x = self.deconv(x)

        return x

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32, class_num=10):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim + self.class_num, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size[0] // 4) * (self.input_size[1] // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.conv(x)
        x = x.view(-1, 128 * (self.input_size[0] // 4) * (self.input_size[1] // 4))
        x = self.fc(x)

        return x

class CGAN(object):
    def __init__(self, args):
        # parameters
        self.args = args
        self.epoch = args.epoch
        self.sample_epoch = args.sample_epoch
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.stage = args.stage
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.run_name = args.run_name
        self.input_size = args.input_size
        self.z_dim = 100
        self.class_num = 3
        self.sample_num = 3 # self.class_num ** 2

        # load dataset
        self.train_loader, self.val_loader, self.test_loader = dataloader(self.input_size, self.batch_size)
        data = self.train_loader.__iter__().__next__()[0]

        # networks init
        self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size, class_num=self.class_num)
        self.D = discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size, class_num=self.class_num)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()
        
        self.radnet = torch.hub.load("Warvito/radimagenet-models", model="radimagenet_resnet50", verbose=True)
        self.radnet.to('cuda')
        self.radnet.eval()
        
        self.fid = FIDMetric()
        self.ms_ssim = MultiScaleSSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=4)
        self.ssim = SSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=4)
        self.psnr = PSNRMetric(max_val= 1.0)

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')

        # fixed noise & condition
        self.sample_z_ = torch.zeros((self.sample_num, self.z_dim))
        for i in range(self.class_num):
            self.sample_z_[i] = torch.rand(1, self.z_dim)
        # for i in range(self.class_num):
        #     self.sample_z_[i*self.class_num] = torch.rand(1, self.z_dim)
        #     for j in range(1, self.class_num):
        #         self.sample_z_[i*self.class_num + j] = self.sample_z_[i*self.class_num]

        temp = torch.zeros((self.class_num, 1))
        for i in range(self.class_num):
            temp[i, 0] = i

        temp_y = torch.zeros((self.sample_num, 1))
        temp_y = temp
        # for i in range(self.class_num):
        #     temp_y[i*self.class_num: (i+1)*self.class_num] = temp

        self.sample_y_ = torch.zeros((self.sample_num, self.class_num)).scatter_(1, temp_y.type(torch.LongTensor), 1)
        if self.gpu_mode:
            self.sample_z_, self.sample_y_ = self.sample_z_.cuda(), self.sample_y_.cuda()
            # self.sample_z_ = self.sample_z_.cuda()
        
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
        
    def train(self):
        
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter, (x_, y_, snp_) in enumerate(self.train_loader):
                if iter == self.train_loader.dataset.__len__() // self.batch_size:
                    break

                z_ = torch.rand((self.batch_size, self.z_dim))
                # y_ = y_.unsqueeze(-1)
                # y_ = (y_ / 2.).unsqueeze(-1)
                # snp_ = snp_ / 2.
                # y_vec_ = y_
                # y_vec_ = torch.cat([y_, snp_], 1)

                y_vec_ = torch.zeros((self.batch_size, self.class_num)).scatter_(1, y_.type(torch.LongTensor).unsqueeze(1), 1) #(64, 3)
                y_fill_ = y_vec_.unsqueeze(2).unsqueeze(3).expand(self.batch_size, self.class_num, self.input_size[0], self.input_size[1]) # (64, 3, 224, 224)

                if self.gpu_mode:
                    x_, z_, y_vec_, y_fill_ = x_.cuda(), z_.cuda(), y_vec_.cuda(), y_fill_.cuda()

                # update D network
                self.D_optimizer.zero_grad()

                D_real = self.D(x_, y_fill_)
                D_real_loss = self.BCE_loss(D_real, self.y_real_)

                G_ = self.G(z_, y_vec_)
                D_fake = self.D(G_, y_fill_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

                D_loss = D_real_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_, y_vec_)
                D_fake = self.D(G_, y_fill_)
                G_loss = self.BCE_loss(D_fake, self.y_real_)
                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward()
                self.G_optimizer.step()

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.train_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item()))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            wandb.log({"D loss": D_loss.item(), 
                       "D loss real": D_real_loss.item(),
                       "D loss fake": D_fake_loss.item(),
                       "G loss": G_loss.item()})

            if ((epoch + 1) % 100) == 0:
                with torch.no_grad():
                    for iter, (x_, y_, snp_) in enumerate(self.test_loader):
                        if iter == self.test_loader.dataset.__len__() // self.batch_size:
                            break
                        y_ = y_.unsqueeze(-1)
                        
                        sample_y_ = torch.zeros(self.batch_size, self.class_num).scatter_(1, y_.type(torch.LongTensor), 1)
                        sample_z_ = torch.rand((self.batch_size, self.z_dim))
                        if self.gpu_mode:
                            x_, sample_z_, sample_y_ = x_.cuda(), sample_z_.cuda(), sample_y_.cuda()

                        samples = self.G(sample_z_, sample_y_)                
                            
                        if self.gpu_mode:
                            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
                        else:
                            samples = samples.data.numpy().transpose(0, 2, 3, 1)           
                                    
                    wandb.log({"sampled_images": [wandb.Image(samples[0])]})   
                                                   
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        # utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '/' + self.run_name,
                                #  self.epoch)
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.model_name, self.run_name), self.model_name)

    def visualize_results(self, epoch, fix=False):
        if self.stage == 'sample':
            self.load()
            
        self.G.eval()
        
        path = os.path.join(self.result_dir, self.model_name, self.run_name, 'samples', self.args.ts)
        os.makedirs(path + '/' + 'AD')
        os.makedirs(path + '/' + 'MCI')
        os.makedirs(path + '/' + 'CN')
        
        fid_radnet = []
        fid_imgnet = []
        ms_ssim_total = []
        ssim_total = []
        psnr_total = []
        
        pfw.set_config(batch_size=self.batch_size, device='cuda')
        
        for idx in range(epoch):
            synth_features = []
            real_features = []
            fid_scores = []
            ms_ssim_scores = []
            ssim_scores = []
            psnr_scores = []
            
            for iter, (x_, y_, snp_) in enumerate(self.test_loader):
                if iter == self.test_loader.dataset.__len__() // self.batch_size:
                    break
                y_ = y_.unsqueeze(-1)
                # y_ = (y_ / 2.).unsqueeze(-1) 
                # snp_ = snp_ / 2.
            
                # image_frame_dim = int(np.floor(np.sqrt(self.sample_num)))

                if fix:
                    """ fixed noise """
                    samples = self.G(self.sample_z_, self.sample_y_)
                else:
                    """ random noise """
                    # sample_y_ = y_
                    # sample_y_ = torch.cat([snp_, y_], 1)
                    sample_y_ = torch.zeros(self.batch_size, self.class_num).scatter_(1, y_.type(torch.LongTensor), 1)
                    sample_z_ = torch.rand((self.batch_size, self.z_dim))
                    if self.gpu_mode:
                        x_, sample_z_, sample_y_ = x_.cuda(), sample_z_.cuda(), sample_y_.cuda()

                    samples = self.G(sample_z_, sample_y_)

                x_ = (x_ + 1.0) / 2.0
                samples = (samples + 1.0) / 2.0
                samples = samples.type(torch.float32)

                psnr_scores.append(self.psnr(samples, x_))
                
                fid_score = pfw.fid(samples, x_)
                fid_scores.append(fid_score)
                
                real_eval_feats = self.get_features(x_)
                real_features.append(real_eval_feats)
                
                synth_eval_feats = self.get_features(samples)
                synth_features.append(synth_eval_feats)
                
                idx_pairs = list(combinations(range(self.batch_size), 2))
                for idx_a, idx_b in idx_pairs:
                    ms_ssim_scores.append(self.ms_ssim(samples[[idx_a]], samples[[idx_b]]))
                    ssim_scores.append(self.ssim(samples[[idx_a]], samples[[idx_b]]))
                
                if self.gpu_mode:
                    samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
                else:
                    samples = samples.data.numpy().transpose(0, 2, 3, 1)                

                for i in range(self.batch_size):
                    if y_[i].item() == 0:
                        utils.save_images(samples[i], os.path.join(path, "AD") + '/' + 'epoch%04d' % idx + f"_{iter}_{i}.png")
                    elif y_[i].item() == 1:
                        utils.save_images(samples[i], os.path.join(path, "CN") + '/' + 'epoch%03d' % idx + f"_{iter}_{i}.png")
                    elif y_[i].item() == 2:
                        utils.save_images(samples[i], os.path.join(path, "MCI") + '/' + 'epoch%03d' % idx + f"_{iter}_{i}.png")
                # utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                #                   self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')
                    wandb.log({"sampled_images": [wandb.Image(samples[i])]})
            
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
        
    def save(self):
        save_dir = os.path.join(self.save_dir, self.model_name, self.run_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.model_name, self.run_name)
        # save_dir = '/root/proj3/gan_collection/models/mri/CGAN'
        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))