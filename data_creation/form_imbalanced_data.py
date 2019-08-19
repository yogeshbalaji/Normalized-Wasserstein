# Code for forming imbalanced dataset
import os
import sys
import argparse
import os.path as osp
import sample_frac_dicts as sdict
import sample_frac_dicts_uniform as sdict_u
from random import shuffle
import math


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def _find_classes(dir, frac_dict):
    """
    Finds the class folders in a dataset.

    Args:
        dir (string): Root directory path.

    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

    Ensures:
        No class is a subdirectory of another.
    """
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if (d.is_dir() and (d.name in frac_dict))]
    else:
        classes = [d for d in os.listdir(dir) if (os.path.isdir(os.path.join(dir, d)) and (d in frac_dict))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, frac_dict, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)

    for target in sorted(class_to_idx.keys()):

        if target not in frac_dict:
            continue

        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        sample_classwise = []
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    # Removing data root
                    path = path.split('/')[-4:]
                    path = '/'.join(path)

                    item = (path, class_to_idx[target])
                    sample_classwise.append(item)
                    # images.append(item)

        shuffle(sample_classwise)
        num_samples = len(sample_classwise)
        num_to_sample = int(math.floor(frac_dict[target] * num_samples))

        for i in range(num_to_sample):
            images.append(sample_classwise[i])

    return images


def dump_dataset(samples, out_file):
    f = open(out_file, 'w')
    for sample in samples:
        msg = '{}, {}\n'.format(sample[0], int(sample[1]))
        f.write(msg)
    f.close()

def form_dataset():

    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', required=True, help='path to source dataset')
    parser.add_argument('--target-root', required=True, help='path to target dataset')
    parser.add_argument('--dataset', required=True, help='dataset, has to be MNIST or VISDA')
    parser.add_argument('--out-path', required=True, help='path where datasets have to be dumped')
    parser.add_argument('--num-modes', default=10, type=int, help='number of modes')
    parser.add_argument('--run-id', default=1, type=int, help='run id')
    parser.add_argument('--uniform', action='store_true')
    args = parser.parse_args()

    # Forming out path
    args.out_path = osp.join(args.out_path, '{}mode'.format(args.num_modes), 'run_{}'.format(args.run_id))
    if not osp.exists(args.out_path):
        os.mkdir(args.out_path)

    assert args.dataset == 'MNIST' or args.dataset == 'VISDA'

    if args.dataset == 'MNIST':
        source_train_root = osp.join(args.source_root, 'trainset')
        source_val_root = osp.join(args.source_root, 'testset')
        target_train_root = osp.join(args.target_root, 'trainset')
        target_val_root = osp.join(args.target_root, 'testset')

        if args.num_modes == 3:
            if args.uniform:
                source_frac_dict = sdict_u._MNIST_m3
                target_frac_dict = sdict_u._MNISTM_m3
            else:
                source_frac_dict = sdict._MNIST_m3
                target_frac_dict = sdict._MNISTM_m3
        elif args.num_modes == 5:
            if args.uniform:
                source_frac_dict = sdict_u._MNIST_m5
                target_frac_dict = sdict_u._MNISTM_m5
            else:
                source_frac_dict = sdict._MNIST_m5
                target_frac_dict = sdict._MNISTM_m5
        elif args.num_modes == 10:
            if args.uniform:
                source_frac_dict = sdict_u._MNIST_m10
                target_frac_dict = sdict_u._MNISTM_m10
            else:
                source_frac_dict = sdict._MNIST_m10
                target_frac_dict = sdict._MNISTM_m10
        else:
            raise ValueError('Enter a valid number of modes [ 3 | 5 | 10 ]')

    else:
        # TODO: fill visda
        pass

    classes, class_to_idx = _find_classes(source_train_root, source_frac_dict)
    print('Class to index mapping ...')
    print(class_to_idx)

    for key in class_to_idx.keys():
        assert key in source_frac_dict
        assert key in target_frac_dict

    print('Forming datasets ...')
    source_train_dataset = make_dataset(source_train_root, class_to_idx, source_frac_dict, extensions=IMG_EXTENSIONS)
    source_val_dataset = make_dataset(source_val_root, class_to_idx, source_frac_dict, extensions=IMG_EXTENSIONS)

    target_train_dataset = make_dataset(target_train_root, class_to_idx, target_frac_dict, extensions=IMG_EXTENSIONS)
    target_val_dataset = make_dataset(target_val_root, class_to_idx, target_frac_dict, extensions=IMG_EXTENSIONS)

    print('Dumping datasets ...')
    # Dumping datasets
    dump_dataset(source_train_dataset, osp.join(args.out_path, 'source_train.txt'))
    dump_dataset(source_val_dataset, osp.join(args.out_path, 'source_val.txt'))
    dump_dataset(target_train_dataset, osp.join(args.out_path, 'target_train.txt'))
    dump_dataset(target_val_dataset, osp.join(args.out_path, 'target_val.txt'))


if __name__ == '__main__':
    form_dataset()

