from __future__ import print_function
import argparse
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
import data
import trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1024,  help='input batch size')
    parser.add_argument('--nz', type=int, default=2, help='size of the latent z vector')
    parser.add_argument('--n-out', type=int, default=2, help='size of the latent z vector')
    parser.add_argument('--nmodes', type=int, default=9, help='size of the latent z vector')
    parser.add_argument('--nepochs', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--pi_update_steps', type=int, default=1, help='update pi per x iterations')
    parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
    parser.add_argument('--lrPi', type=float, default=0.01, help='learning rate for Generator, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
    parser.add_argument('--Diters', type=int, default=10, help='number of D iters per each G iter')
    parser.add_argument('--save-root', default='results', help='Path to store samples and models')
    parser.add_argument('--experiment', required=True, choices=["gaussian", "mog"],
                        help='Name of experiment | gaussian, mog')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
    args = parser.parse_args()
    print(args)
    return args


def main(args):

    # Seeds
    args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    cudnn.benchmark = True
    np.random.seed(1000)

    # Data loader
    data_config = {}
    if args.experiment == "gaussian":
        num_modes = args.nmodes
        num_grids = num_modes
        mode_prob = [0.95, 0.05]
        data_config["rad"] = 4
    else:
        num_modes = args.nmodes
        num_grids = int(np.sqrt(num_modes))
        num_modes = num_grids*num_grids

        mode_prob = [(num_modes - i) for i in range(num_modes)]
        mode_prob = np.array(mode_prob).astype(np.float32)
        mode_prob = mode_prob / np.sum(mode_prob)
        data_config["grid_spacing"] = 2

    data_config["num_modes"] = num_modes
    data_config["num_grids"] = num_grids
    data_config["mode_prob"] = mode_prob

    if args.experiment == "gaussian":
        dataloader = data.GaussianData(args, data_config)
    else:
        dataloader = data.GaussianDataGrid(args, data_config)

    _trainer = trainer.GANTrainer(args, dataloader, data_config)
    _trainer.train()


if __name__ == '__main__':
    args = parse_args()
    main(args)