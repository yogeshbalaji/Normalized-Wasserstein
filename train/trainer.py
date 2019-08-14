import torch
import torch.nn as nn
import datasets
import torchvision.transforms as T
import models
import torch.optim as optim
import torch.nn.functional as F
import utils
import numpy as np
from torch.utils.data import DataLoader


class Trainer(object):
    def __init__(self, args, data_samples, nclasses=10):

        self.args = args

        ###############################
        # Create dataloaders

        dataset_mean = (0.5, 0.5, 0.5)
        dataset_std = (0.5, 0.5, 0.5)

        source_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=dataset_mean,
                        std=dataset_std)
        ])
        target_transform_train = T.Compose([
            T.RandomCrop((28)),
            T.ToTensor(),
            T.Normalize(mean=dataset_mean,
                        std=dataset_std)
        ])
        target_transform_test = T.Compose([
            T.CenterCrop((28)),
            T.ToTensor(),
            T.Normalize(mean=dataset_mean,
                        std=dataset_std)
        ])

        dat_s_train = datasets.ImageFolder(data_samples['source_train'],
                                                  transform=source_transform)
        dat_s_val = datasets.ImageFolder(data_samples['source_val'],
                                                  transform=source_transform)
        dat_t_train = datasets.ImageFolder(data_samples['target_train'],
                                                  transform=target_transform_train)
        dat_t_val = datasets.ImageFolder(data_samples['target_val'],
                                                transform=target_transform_test)

        self.s_trainloader = DataLoader(dataset=dat_s_train,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        drop_last=True,
                                        num_workers=args.workers)
        self.s_valloader = DataLoader(dataset=dat_s_val,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      drop_last=False,
                                      num_workers=args.workers)
        self.t_trainloader = DataLoader(dataset=dat_t_train,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        drop_last=True,
                                        num_workers=args.workers)
        self.t_valloader = DataLoader(dataset=dat_t_val,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      drop_last=False,
                                      num_workers=args.workers)

        ## For NW, use classwise dataloader
        data_samples_classwise = utils.form_samples_classwise(data_samples['source_train'],
                                                              nclasses)
        self.s_trainloader_classwise = [
            DataLoader(
                datasets.ImageFolder(data_samples_classwise[cl],
                                     transform=source_transform),
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=args.workers) for cl in range(nclasses)
        ]

        self.s_classwise_iterators = []
        for i in range(len(self.s_trainloader_classwise)):
            self.s_classwise_iterators.append(iter(self.s_trainloader_classwise[i]))

        self.nclasses = nclasses

        ###############################
        # Create models

        self.netF = models._netF().cuda()
        self.netC = models._netC(self.nclasses).cuda()
        if args.alg == 'wasserstein' or args.alg == 'NW':
            self.netD = models._netD_wasserstein().cuda()
        else:
            self.netD = models._netD().cuda()

        # Create optimizers
        if args.adam:
            self.optimizerF = optim.Adam(self.netF.parameters(), lr=args.lr, betas=(0.5, 0.999))
            self.optimizerC = optim.Adam(self.netC.parameters(), lr=args.lr, betas=(0.5, 0.999))
            self.optimizerD = optim.Adam(self.netD.parameters(), lr=args.lr, betas=(0.5, 0.999))
            if args.alg == 'NW':
                self.pi = nn.Parameter(torch.FloatTensor(nclasses).fill_(1.0 / nclasses).cuda())
                self.optimizerPi = optim.Adam(iter([self.pi]), lr=args.lrPi, betas=(0.5, 0.999))
        else:
            self.optimizerF = optim.SGD(self.netF.parameters(), lr=args.lr, momentum=0.9)
            self.optimizerC = optim.SGD(self.netC.parameters(), lr=args.lr, momentum=0.9)
            self.optimizerD = optim.SGD(self.netD.parameters(), lr=args.lr, momentum=0.9)
            if args.alg == 'NW':
                self.pi = nn.Parameter(torch.FloatTensor(nclasses).fill_(1.0 / nclasses).cuda())
                self.optimizerPi = optim.SGD(iter([self.pi]), lr=args.lrPi)

    def _zero_grad(self):
        self.optimizerF.zero_grad()
        self.optimizerC.zero_grad()
        self.optimizerD.zero_grad()
        if self.args.alg == 'NW':
            self.optimizerPi.zero_grad()

    def sample_classwise(self, class_id):
        try:
            batch = next(self.s_classwise_iterators[class_id])
        except StopIteration:
            self.s_classwise_iterators[class_id] = iter(self.s_trainloader_classwise[class_id])
            batch = next(self.s_classwise_iterators[class_id])
        return batch

    def test(self):

        self.netF.eval()
        self.netC.eval()

        ### Validating source domain
        correct = [0] * self.nclasses
        total = [0] * self.nclasses

        for i, s_data in enumerate(self.s_valloader):
            inputs, labels = s_data
            inputs = inputs.cuda()
            labels = labels.cuda()

            with torch.no_grad():
                logits = self.netC(self.netF(inputs))
            _, pred = torch.max(logits, dim=1)

            for j in range(labels.size(0)):
                lab = labels[j]
                correct[lab] += (pred[j] == labels[j]).sum()
                total[lab] += 1

        train_acc_classwise = np.array([float(correct[j] * 100)/total[j] for j in range(self.nclasses)])
        train_acc_mean = train_acc_classwise.mean()

        print('Source validation accuracy: {}'.format(train_acc_mean))
        print(train_acc_classwise)

        ### Validating target domain
        correct = [0] * self.nclasses
        total = [0] * self.nclasses

        for i, t_data in enumerate(self.t_valloader):
            inputs, labels = t_data
            inputs = inputs.cuda()
            labels = labels.cuda()

            with torch.no_grad():
                logits = self.netC(self.netF(inputs))
            _, pred = torch.max(logits, dim=1)

            for j in range(labels.size(0)):
                lab = labels[j]
                correct[lab] += (pred[j] == labels[j]).item()
                total[lab] += 1

        train_acc_classwise = np.array([float(correct[j] * 100) / total[j] for j in range(self.nclasses)])
        train_acc_mean = train_acc_classwise.mean()

        print('Target validation accuracy: {}'.format(train_acc_mean))
        print(train_acc_classwise)
        print('#########################################')



    def train(self):

        curr_iter = 0

        for epoch in range(self.args.epochs):
            print('Starting epoch {} ...'.format(epoch))

            self.netF.train()
            self.netC.train()
            self.netD.train()

            for i, (s_data, t_data) in enumerate(zip(self.s_trainloader, self.t_trainloader)):
                curr_iter += 1
                s_inputs, s_labels = s_data
                t_inputs, _ = t_data

                s_inputs = s_inputs.cuda()
                s_labels = s_labels.cuda()
                t_inputs = t_inputs.cuda()

                # Domain labels
                s_labels_dom = torch.LongTensor(s_inputs.size(0)).fill_(0).cuda()
                t_labels_dom = torch.LongTensor(s_inputs.size(0)).fill_(1).cuda()

                self._zero_grad()
                s_feat = self.netF(s_inputs)
                s_logits_cls = self.netC(s_feat)
                class_loss = F.cross_entropy(s_logits_cls, s_labels)

                if self.args.alg != 'sourceonly':
                    t_feat = self.netF(t_inputs)
                    t_logits_dom = self.netD(t_feat)

                    if self.args.alg == 'NW':
                        for mode in range(self.nclasses):
                            s_inputs_classwise, s_lab_cl = self.sample_classwise(mode)
                            s_inputs_classwise = s_inputs_classwise.cuda()

                            if mode == 0:
                                s_logits_dom = F.softmax(self.pi, dim=0)[mode] * \
                                               self.netD(self.netF(s_inputs_classwise))
                            else:
                                s_logits_dom = s_logits_dom + F.softmax(self.pi, dim=0)[mode] * \
                                               self.netD(self.netF(s_inputs_classwise))
                    else:
                        s_logits_dom = self.netD(s_feat)


                if self.args.alg == 'dann':

                    s_dom_loss = F.cross_entropy(s_logits_dom, s_labels_dom)
                    t_dom_loss = F.cross_entropy(t_logits_dom, t_labels_dom)
                    dom_loss = (s_dom_loss + t_dom_loss)
                    dom_loss.backward(retain_graph=True)
                    self.optimizerD.step()

                    self._zero_grad()
                    loss_net = class_loss - dom_loss
                    loss_net.backward()
                    self.optimizerC.step()
                    self.optimizerF.step()

                elif self.args.alg == 'wasserstein' or self.args.alg == 'NW':

                    # Default params for Wasserstein loss
                    gamma = 1
                    burn_out = 10000000
                    clamp_val = 0.01
                    critic_iters = 10

                    one = torch.FloatTensor([1])
                    mone = one * -1
                    one = one.cuda()
                    mone = mone.cuda()

                    for p in self.netD.parameters():
                        p.data.clamp_(-clamp_val, clamp_val)
                    self._zero_grad()

                    t_logits_dom.backward(one, retain_graph=True)
                    s_logits_dom.backward(mone, retain_graph=True)
                    dom_loss = t_logits_dom - s_logits_dom
                    self.optimizerD.step()

                    if self.args.alg == 'NW':
                        self.optimizerPi.zero_grad()
                        s_logits_dom.backward(one, retain_graph=True)
                        self.optimizerPi.step()

                    if curr_iter == burn_out:
                        print('Burn out over')

                    if curr_iter % critic_iters == 0 or curr_iter < burn_out:
                        self._zero_grad()

                        if curr_iter < burn_out and curr_iter % critic_iters > 0:
                            class_loss.backward()
                            self.optimizerC.step()
                            self.optimizerF.step()

                        else:
                            (gamma * s_logits_dom).backward(one, retain_graph=True)
                            (gamma * t_logits_dom).backward(mone)
                            class_loss.backward()
                            self.optimizerC.step()
                            self.optimizerF.step()

                elif self.args.alg == 'sourceonly':
                    class_loss.backward()
                    self.optimizerC.step()
                    self.optimizerF.step()


                # Logging the results
                if i % 100 == 0:
                    if self.args.alg == 'NW':
                        print('Mode prob')
                        print(F.softmax(self.pi, dim=0))

                    if self.args.alg == 'sourceonly':
                        print('[{}/{}] \t Classification loss: {}'.format(i * len(s_inputs),
                                                                          len(self.t_trainloader.dataset),
                                                                          class_loss.item()))
                    else:
                        print('[{}/{}] \t Classification loss: {}, '
                              'Domain loss: {}'.format(i * len(s_inputs),
                                                        len(self.t_trainloader.dataset),
                                                        class_loss.item(),
                                                        dom_loss.item()))

            self.test()

