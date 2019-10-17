import torchvision.transforms as T
from .folder import ImageFolder
from torch.utils.data import DataLoader
from .utils import read_file, form_samples_classwise
import os.path as osp
import numpy as np


# Datasets for MNIST -> MNIST-M experiments
def form_mnist_dataset(args):

    s_train = read_file(osp.join(args.data_path, 'source_train.txt'), data_root=args.data_root)
    s_val = read_file(osp.join(args.data_path, 'source_val.txt'), data_root=args.data_root)
    t_train = read_file(osp.join(args.data_path, 'target_train.txt'), data_root=args.data_root)
    t_val = read_file(osp.join(args.data_path, 'target_val.txt'), data_root=args.data_root)
    data_samples = {
        'source_train': s_train,
        'source_val': s_val,
        'target_train': t_train,
        'target_val': t_val
    }

    all_labels = np.array([int(s[1]) for s in s_train])
    nclasses = len(np.unique(all_labels))
    print('Number of classes: {}'.format(nclasses))

    dataset_mean = (0.5, 0.5, 0.5)
    dataset_std = (0.5, 0.5, 0.5)

    source_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=dataset_mean,
                    std=dataset_std)
    ])
    target_transform_train = T.Compose([
        T.RandomCrop(28),
        T.ToTensor(),
        T.Normalize(mean=dataset_mean,
                    std=dataset_std)
    ])
    target_transform_test = T.Compose([
        T.CenterCrop(28),
        T.ToTensor(),
        T.Normalize(mean=dataset_mean,
                    std=dataset_std)
    ])

    dat_s_train = ImageFolder(data_samples['source_train'],
                              transform=source_transform)
    dat_s_val = ImageFolder(data_samples['source_val'],
                            transform=source_transform)
    dat_t_train = ImageFolder(data_samples['target_train'],
                              transform=target_transform_train)
    dat_t_val = ImageFolder(data_samples['target_val'],
                            transform=target_transform_test)

    s_trainloader = DataLoader(dataset=dat_s_train,
                               batch_size=args.batch_size,
                               shuffle=True,
                               drop_last=True,
                               num_workers=args.workers)
    s_valloader = DataLoader(dataset=dat_s_val,
                             batch_size=args.batch_size,
                             shuffle=False,
                             drop_last=False,
                             num_workers=args.workers)
    t_trainloader = DataLoader(dataset=dat_t_train,
                               batch_size=args.batch_size,
                               shuffle=True,
                               drop_last=True,
                               num_workers=args.workers)
    t_valloader = DataLoader(dataset=dat_t_val,
                             batch_size=args.batch_size,
                             shuffle=False,
                             drop_last=False,
                             num_workers=args.workers)

    data_samples_classwise = form_samples_classwise(data_samples['source_train'],
                                                          nclasses)
    s_trainloader_classwise = [
        DataLoader(
            ImageFolder(data_samples_classwise[cl],
                        transform=source_transform),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.workers) for cl in range(nclasses)
    ]

    dataloaders = {
        's_train': s_trainloader,
        's_val': s_valloader,
        't_train': t_trainloader,
        't_val': t_valloader,
        's_classwise': s_trainloader_classwise,
        'nclasses': nclasses
    }
    return dataloaders