import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import models.linear as linear
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import utils
import os
from pathlib import Path
import math


class GANTrainer(object):
    def __init__(self, args, dataloader, data_config):

        self.dataloader = dataloader
        self.num_modes = num_modes = data_config["num_modes"]
        self.args = args

        self.save_path = os.path.join(args.save_root, args.experiment)
        Path(self.save_path).mkdir(exist_ok=True, parents=True)
        self.log_file = open('{}/log.txt'.format(self.save_path), 'w')

        # Models
        netG = []
        for mode in range(num_modes):
            netG.append(linear.NetG(args.nz, args.n_out))
            netG[mode].apply(utils.weights_init)
        print('Generator network')
        print(netG)

        netD = linear.NetDMixtureLearnPi(args.n_out, num_modes)
        netD.apply(utils.weights_init)
        print('Discriminator network')
        print(netD)

        pi = nn.Parameter(torch.FloatTensor(num_modes).uniform_().cuda())
        print('Pi')
        print(pi)

        self.inp = torch.FloatTensor(args.batchSize, args.n_out)
        self.noise = []
        for mode in range(num_modes):
            self.noise.append(torch.FloatTensor(args.batchSize, args.nz))
        self.one = torch.FloatTensor([1])
        self.mone = self.one * -1

        netD.cuda()
        for mode in range(num_modes):
            netG[mode].cuda()
            self.noise[mode] = self.noise[mode].cuda()
        self.inp = self.inp.cuda()
        self.one, self.mone = self.one.cuda(), self.mone.cuda()

        self.netG = netG
        self.netD = netD
        self.pi = pi

        # Optimizers
        D_params = self.netD.parameters()
        self.optimizerG = []
        if args.adam:
            self.optimizerD = optim.Adam(D_params, lr=args.lrD, betas=(args.beta1, 0.999))
            for mode in range(num_modes):
                self.optimizerG.append(optim.Adam(self.netG[mode].parameters(),
                                                  lr=args.lrG, betas=(args.beta1, 0.999)))
            self.optimizerPi = optim.Adam(iter([self.pi]), lr=args.lrPi, betas=(args.beta1, 0.999))
        else:
            self.optimizerD = optim.RMSprop(D_params, lr=args.lrD)
            for mode in range(num_modes):
                self.optimizerG.append(optim.RMSprop(self.netG[mode].parameters(), lr=args.lrG))
            self.optimizerPi = optim.RMSprop(iter([self.pi]), lr=args.lrPi)

    def visualize(self, it):
        """
        Module to visualize generated samples
        """
        num_ep = 2
        real_samples = []
        gen_samples = []

        for mode in range(self.num_modes):
            self.netG[mode].eval()
            gen_samples.append([])

        for i in range(num_ep):
            data = self.dataloader.sample(self.args.batchSize)
            real_samples.append(data.numpy())
            for mode in range(self.num_modes):
                self.noise[mode].resize_(self.args.batchSize, self.args.nz).normal_(0, 0.1)
                noisev = self.noise[mode]
                noisev.requires_grad = False

                fake = self.netG[mode](noisev)
                fake = fake.data.cpu().numpy()
                fake = fake[0:int(F.softmax(self.pi, dim=0)[mode].item() * self.args.batchSize)]
                gen_samples[mode].append(fake)

        real_samples = np.concatenate(real_samples)
        plt.scatter(real_samples[:, 0], real_samples[:, 1], c='r', s=1)

        for mode in range(self.num_modes):
            gen_samples[mode] = np.concatenate(gen_samples[mode])
            plt.scatter(gen_samples[mode][:, 0], gen_samples[mode][:, 1], c='b', s=1)

        plt.legend(['Input distribution', 'Generated distribution'])
        Path('{}/plots'.format(self.save_path)).mkdir(exist_ok=True, parents=True)
        plt.savefig('{}/plots/plot_{}.png'.format(self.save_path, it), dpi=400)

        for mode in range(self.num_modes):
            self.netG[mode].train()
        plt.close()

    def assign_points(self, samples, data_means, data_covars):

        samples = np.squeeze(np.transpose(samples))
        for mode in range(self.num_modes):
            mean = np.expand_dims(data_means[mode], 1)
            # covar_inv = np.linalg.pinv(data_covars[mode])
            dist = np.diag(np.matmul(np.transpose(samples - mean), (samples - mean)))
            dist = np.expand_dims(dist, 1)
            if mode == 0:
                dist_all = dist
            else:
                dist_all = np.concatenate((dist_all, dist), axis=1)

        assignments = np.argmin(dist_all, axis=1)
        return assignments

    def evaluate(self, data_means, data_covars, mode_probs):
        # For each sample in the generated data, we assign it to the cluster with minimum Mahanalobis distance
        # With this assignment, mean, covariance and pi estimation error is computed

        num_samples = 2000
        samples_all = []
        for mode in range(self.num_modes):
            self.netG[mode].eval()

        for i in range(int(math.ceil(num_samples / self.args.batchSize))):
            for mode in range(self.num_modes):
                self.noise[mode].resize_(self.args.batchSize, self.args.nz).normal_(0, 0.1)
                noisev = self.noise[mode]  # totally freeze netG
                noisev.requires_grad = False
                fake = self.netG[mode](noisev)
                fake = fake.data.cpu().numpy()
                fake = fake[0:int(F.softmax(self.pi, dim=0)[mode].item() * self.args.batchSize)]
                samples_all.append(fake)

        samples_all = np.concatenate(samples_all, axis=0)
        assignments_all = self.assign_points(samples_all, data_means, data_covars)

        # Evaluation metrics
        pi_est = np.array(
            [float(np.sum(assignments_all == mode)) / len(assignments_all) for mode in range(self.num_modes)])
        pi_error = np.mean((pi_est - mode_probs) ** 2) / np.mean(mode_probs ** 2)

        print('Estimated pi')
        print(pi_est)

        # Mean and covariance error
        mean_error = 0
        cov_error = 0
        for mode in range(self.num_modes):
            indices = np.nonzero(assignments_all == mode)
            indices = indices[0]
            if len(indices) > 0:
                samples_mode = samples_all[indices, ::]
                sample_mean = np.mean(samples_mode, axis=0)
                mean_error += np.mean((sample_mean - data_means[mode]) ** 2)

                sample_cov = np.cov(np.transpose(samples_mode))
                cov_error += np.mean((sample_cov - data_covars[mode]) ** 2)

        mean_error = mean_error / self.num_modes
        cov_error = cov_error / self.num_modes

        return mean_error, cov_error, pi_error

    def train(self):

        str_to_log = 'Disc loss, Gen loss\n'
        self.log_file.write(str_to_log)

        gen_iterations = 0
        for epoch in range(self.args.nepochs):
            i = 0
            for mode in range(self.num_modes):
                self.netG[mode].train()

            while i < 500: # Every 500 discriminator steps treated as 1 epoch
                # D update
                for p in self.netD.parameters():
                    p.requires_grad = True

                if gen_iterations < 25 or gen_iterations % 500 == 0:
                    Diters = 20
                else:
                    Diters = self.args.Diters

                if epoch == self.args.nepochs - 1:
                    # We measure the wasserstein distance in the last epoch,
                    # so discriminator is trained to optimality
                    Diters = 1000

                j = 0
                while j < Diters:
                    j += 1

                    # clamp parameters to a cube
                    for p in self.netD.parameters():
                        p.data.clamp_(self.args.clamp_lower, self.args.clamp_upper)

                    data = self.dataloader.sample(self.args.batchSize)
                    i += 1

                    # train with real
                    real_cpu = data
                    self.netD.zero_grad()
                    batch_size = real_cpu.size(0)

                    real_cpu = real_cpu.cuda()
                    self.inp.resize_as_(real_cpu).copy_(real_cpu)

                    errD_real = self.netD(self.inp, real_flag=True)
                    errD_real.backward(self.mone)

                    # train with fake
                    fake_list = []
                    for mode in range(self.num_modes):
                        self.noise[mode].resize_(self.args.batchSize, self.args.nz).normal_(0, 0.1)
                        noisev = self.noise[mode]  # totally freeze netG
                        noisev.requires_grad = False
                        fake_list.append(self.netG[mode](noisev).data)

                    errD_fake = self.netD(fake_list, self.pi, real_flag=False)
                    errD_fake.backward(self.one)
                    errD = errD_real - errD_fake
                    self.optimizerD.step()

                # G updates
                for p in self.netD.parameters():
                    p.requires_grad = False

                for mode in range(self.num_modes):
                    self.netG[mode].zero_grad()
                self.optimizerPi.zero_grad()

                fake_list = []
                for mode in range(self.num_modes):
                    self.noise[mode].resize_(self.args.batchSize, self.args.nz).normal_(0, 0.1)
                    noisev = self.noise[mode]
                    fake_list.append(self.netG[mode](noisev))

                errG = self.netD(fake_list, self.pi, real_flag=False)
                errG.backward(self.mone)

                for mode in range(self.num_modes):
                    self.optimizerG[mode].step()

                if gen_iterations % self.args.pi_update_steps == 0:
                    self.optimizerPi.step()

                gen_iterations += 1
                if gen_iterations % 10 == 0:
                    print('[%d/%d][%d/500][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                          % (epoch, self.args.nepochs, i, gen_iterations,
                             errD.item(), errG.item(), errD_real.item(), errD_fake.item()))
                    str_to_log = '{}\t{}\n'.format(errD.item(), errD.item())
                    self.log_file.write(str_to_log)

                if epoch == self.args.nepochs - 1:
                    str_to_log = 'Final Wasserstein distance: {}'.format(errD.item())
                    print(str_to_log)
                    self.log_file.write(str_to_log)

                if gen_iterations % 200 == 0:
                    # validate()
                    print('Mode probs')
                    print(torch.sort(F.softmax(self.pi, dim=0)))
                    self.visualize(epoch)

                    (mean_error, cov_error, pi_error) = self.evaluate(self.dataloader.data_means,
                                                                      self.dataloader.data_covars,
                                                                      self.dataloader.mode_prob)
                    print('Mean error: {}'.format(mean_error))
                    print('Covariance error: {}'.format(cov_error))
                    print('Pi error: {}'.format(pi_error))

        self.log_file.close()