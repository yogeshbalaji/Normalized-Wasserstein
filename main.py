"""
Main script for models
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
from train import Trainer


def parse_arguments():
    """Command line parse."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type= str, required=True)
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--exp', type= str, default= 'MNIST', help = 'Choose an experiment| MNIST, VISDA')
    parser.add_argument('--alg', type=str, default='dann', help='Choose an algorithm to train the model.')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers in dataloader')
    parser.add_argument('--lr', type= float, default=0.0002, help= 'Learning rate.')
    parser.add_argument('--lrPi', type= float, default= 0.0005, help= 'Learning rate.')
    parser.add_argument('--adam', action='store_true', help='use adam or not')
    parser.add_argument('--log_interval', type=int, default=25, help='number of iterations to log')
    parser.add_argument('--saver_root', type=str, default='results/MNIST', help='saver path')
    return parser.parse_args()


def main(args):

    assert args.alg in ['sourceonly', 'dann', 'wasserstein', 'NW']

    trainer = Trainer(args)
    trainer.train()

    print('Final accuracy')
    trainer.test()


if __name__ == '__main__':
    main(parse_arguments())
