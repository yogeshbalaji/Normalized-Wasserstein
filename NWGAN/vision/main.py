"""WGAN-GP ResNet for CIFAR-10"""

import os, sys
sys.path.append(os.getcwd())
import tflib as lib
import locale
locale.setlocale(locale.LC_ALL, '')
import argparse
from trainers import trainer
import json
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-path', type=str, required=True, help='Config path')
    return parser.parse_args()


def main(args):

    with open(args.cfg_path, "r") as fp:
        configs = json.load(fp)
    configs = utils.ConfigMapper(configs)

    utils.mkdir_p(configs.log_dir)
    utils.mkdir_p(configs.log_dir + '/models')

    lib.print_model_settings(locals().copy())
    trainer.train(configs)


if __name__ == '__main__':
    args = parse_args()
    main(args)