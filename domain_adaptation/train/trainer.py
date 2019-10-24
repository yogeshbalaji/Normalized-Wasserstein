import torch
import torch.nn as nn
import datasets
import models
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pathlib import Path


class Trainer(object):
    def __init__(self, args):

        self.args = args
        Path(args.saver_root).mkdir(parents=True, exist_ok=True)

        if args.exp == 'MNIST':
            self.log('Running MNIST -> MNIST-M')
            dataloders = datasets.form_mnist_dataset(args)
        elif args.exp == 'VISDA':
            # TODO: Include VISDA
            pass

        self.s_trainloader = dataloders['s_train']
        self.s_valloader = dataloders['s_val']
        self.t_trainloader = dataloders['t_train']
        self.t_valloader = dataloders['t_val']
        self.s_trainloader_classwise = dataloders['s_classwise']
        nclasses = self.nclasses = dataloders['nclasses']

        self.s_classwise_iterators = []
        for i in range(len(self.s_trainloader_classwise)):
            self.s_classwise_iterators.append(iter(self.s_trainloader_classwise[i]))

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

    def log(self, message):
        print(message)
        message = message + '\n'
        f = open("{}/log.txt".format(self.args.saver_root), "a+")
        f.write(message)
        f.close()

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

        # Validating source domain
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

        msg = 'Source validation accuracy: {}'.format(train_acc_mean)
        self.log(msg)
        print('Classwise accuracy')
        print(train_acc_classwise)

        # Validating target domain
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

        msg = 'Target validation accuracy: {}'.format(train_acc_mean)
        self.log(msg)
        print('Classwise accuracy')
        print(train_acc_classwise)
        print('#########################################')

    def train(self):
        curr_iter = 0
        for epoch in range(self.args.epochs):
            self.log('Starting epoch {} ...'.format(epoch))
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

                    self._zero_grad()
                    if curr_iter % critic_iters > 0:
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
                if i % self.args.log_interval == 0:
                    if self.args.alg == 'NW':
                        print('Mode prob')
                        print(F.softmax(self.pi, dim=0))

                    if self.args.alg == 'sourceonly':
                        self.log('[{}/{}] \t Classification loss: {}'.format(i * len(s_inputs),
                                                                             len(self.t_trainloader.dataset),
                                                                             class_loss.item()))
                    else:
                        self.log('[{}/{}] \t Classification loss: {}, '
                                 'Domain loss: {}'.format(i * len(s_inputs),
                                                          len(self.t_trainloader.dataset),
                                                          class_loss.item(),
                                                          dom_loss.item()))

            self.test()
