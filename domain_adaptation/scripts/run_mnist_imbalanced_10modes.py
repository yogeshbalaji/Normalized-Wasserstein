# Script for running imbalanced MNIST -> MNIST-M experiments

import os
import os.path as osp
import argparse
from multiprocessing import Pool


DATA_ROOT = '/vulcan/scratch/yogesh22/data/digits_full'
DATA_PATH = 'data/MNIST-MNISTM/'
SAVE_ROOT = 'results/MNIST-MNISTM'


def run_process(config):
    print('Running gpu {}, {} ...'.format(config[0], config[1]))
    cmd = 'CUDA_VISIBLE_DEVICES={} python main.py --data-path {} --data-root {} --exp MNIST --alg {} ' \
          '--saver_root {} --adam --epochs 100'.format(config[0], config[1], DATA_ROOT, config[2], config[3])
    os.system(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngpus', required=True, type=int, help='number of GPUs to use')
    parser.add_argument('--alg', type=str, default='NW', help='Choose an algorithm to train the model.')
    args = parser.parse_args()

    mode_list = ['10mode']
    run_list = ['run_1', 'run_2', 'run_3']
    config_list = []
    counter = 0
    for mode in mode_list:
        for run in run_list:
            data_path = osp.join(DATA_PATH, mode)
            saver_root = osp.join(SAVE_ROOT, args.alg, mode, run)
            config = (counter % args.ngpus, data_path, args.alg, saver_root)
            config_list.append(config)
            counter += 1

    pool = Pool(processes=args.ngpus)
    pool.map(run_process, config_list, chunksize=1)


if __name__ == '__main__':
    main()
