import torch
import numpy as np


class GaussianData(object):
    """
    Class for infinite gaussian data loader
    """
    def __init__(self, args, data_config):
        # mode_prob = [0.95, 0.05]
        # rad = 4

        self.num_modes = data_config["num_modes"]
        self.mode_prob = data_config["mode_prob"]
        self.rad = data_config["radius"]
        print('Mode probabilities')
        print(self.mode_prob)

        # Random affine transformation applied to Gaussians
        self.transformation = []
        for mode in range(self.num_modes):
            tx = np.random.randn(args.n_out, args.nz)
            self.transformation.append(tx)

        # Data means
        self.data_means = []
        for mode in range(self.num_modes):
            mean = np.array([self.rad * np.cos(2 * np.pi * (float(mode) / self.num_modes)),
                             self.rad * np.sin(2 * np.pi * (float(mode) / self.num_modes))])
            self.data_means.append(mean)

    def sample(self, batch_size):

        sample_dist = np.random.multinomial(batch_size, self.mode_prob, size=1)
        sample_dist = np.squeeze(sample_dist, 0)

        samples = []
        for mode in range(self.num_modes):
            mean = self.data_means[mode]
            covar = np.matmul(self.transformation[mode], np.transpose(self.transformation[mode]))
            out = np.random.multivariate_normal(mean, covar, sample_dist[mode]).astype(np.float32)
            samples.append(out)
        samples = np.concatenate(samples)
        index_list = np.random.permutation(batch_size)
        samples = samples[index_list, :]

        out_torch = torch.from_numpy(samples)
        return out_torch


class GaussianDataGrid(object):
    """
    Class for infinite gaussian data loader in a grid
    """
    def __init__(self, args, data_config):
        # [(self.args.num_modes - i) for i in range(self.args.num_modes)]
        # self.mode_prob = np.array(self.mode_prob).astype(np.float32)
        # self.mode_prob = self.mode_prob / np.sum(self.mode_prob)
        # grid_spacing = 2

        self.transformation = []
        self.num_modes = data_config["num_modes"]
        self.mode_prob = data_config["mode_prob"]
        grid_spacing = data_config["grid_spacing"]
        self.num_grids = data_config["num_grids"]

        for mode in range(self.num_modes):
            tx = np.random.randn(args.n_out, args.nz) * 0.15
            self.transformation.append(tx)

        print('Transformation matrix')
        print(self.transformation)

        print('Mode probabilities')
        print(self.mode_prob)

        # Creating grid data
        self.data_means = []
        self.data_covars = []
        for mode in range(self.num_modes):
            y_index = int(mode / self.num_grids)
            x_index = mode % self.num_grids

            mean = np.array([x_index * grid_spacing, y_index * grid_spacing])
            self.data_means.append(mean)
            covar = np.matmul(self.transformation[mode], np.transpose(self.transformation[mode]))
            self.data_covars.append(covar)

    def sample(self, batch_size):

        sample_dist = np.random.multinomial(batch_size, self.mode_prob, size=1)
        sample_dist = np.squeeze(sample_dist, 0)

        samples = []
        for mode in range(self.num_modes):
            mean = self.data_means[mode]
            covar = self.data_covars[mode]
            out = np.random.multivariate_normal(mean, covar, sample_dist[mode]).astype(np.float32)
            samples.append(out)

        samples = np.concatenate(samples)
        index_list = np.random.permutation(batch_size)
        samples = samples[index_list, :]
        out_torch = torch.from_numpy(samples)
        return out_torch
