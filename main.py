"""
Main script for models
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import argparse, sys, os
import os.path as osp
import utils
from train import Trainer
import numpy as np

def parse_arguments():
    """Command line parse."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type= str, required=True)
    parser.add_argument('--exp', type= str, default= 'MNIST', help = 'Choose an experiment| MNIST, VISDA')
    parser.add_argument('--alg', type=str, default='dann', help='Choose an algorithm to train the model.')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers in dataloader')
    parser.add_argument('--lr', type= float, default= 0.0002, help= 'Learning rate.')
    parser.add_argument('--lrPi', type= float, default= 0.0005, help= 'Learning rate.')
    parser.add_argument('--adam', action='store_true', help='use adam or not')
    return parser.parse_args()


def main(args):

    assert args.alg in ['sourceonly', 'dann', 'wasserstein', 'NW']

    if args.exp == 'MNIST':

        # Forming data_samples
        s_train = utils.read_file(osp.join(args.data_path, 'source_train.txt'))
        s_val = utils.read_file(osp.join(args.data_path, 'source_val.txt'))
        t_train = utils.read_file(osp.join(args.data_path, 'target_train.txt'))
        t_val = utils.read_file(osp.join(args.data_path, 'target_val.txt'))
        data_samples = {
            'source_train': s_train,
            'source_val': s_val,
            'target_train': t_train,
            'target_val': t_val
        }
        all_labels = np.array([int(s[1]) for s in s_train])
        nclasses = len(np.unique(all_labels))
        print('Number of classes: {}'.format(nclasses))
    else:
        # TODO: Fill in for VISDA
        pass

    trainer = Trainer(args, data_samples, nclasses)
    trainer.train()

    print('Final accuracy')
    trainer.test()


if __name__ == '__main__':
    main(parse_arguments())
