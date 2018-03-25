

import numpy as np
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Batcher:

    def __init__(self, inds_docs_now, labels, batchsize=None, batchtype='iterated', ignore=None, lenbound=None, ignore_y=-1):

        self.inds_docs_now = inds_docs_now
        self.labels = labels
        self.batchsize = batchsize
        self.batchtype = batchtype
        self.ignore = ignore
        self.lenbound = lenbound
        self.ignore_y = ignore_y

        self.n_data = len(self.labels)
        self.n_class = self.labels.max() + 1
        self.Y = np.zeros((self.n_data, self.n_class), dtype=np.float32)
        self.Y[np.arange(self.n_data), labels] = 1
        self.lens = [len(self.inds_docs_now[indnow]) for indnow in np.arange(self.n_data)]
        self.maxlen = np.max(self.lens)
        self.maxind = np.max([np.max(docnow) if len(docnow) > 0 else 0 for docnow in self.inds_docs_now])

        self.iternow = 0
        self.isfinished = False

        if self.ignore is None:
            self.ignore = self.maxind + 1

    def reset_iter(self):

        self.iternow = 0
        self.isfinished = False

    def get_batch_byinds(self, indsnow):

        batchsize = len(indsnow)

        maxlen = np.max([len(self.inds_docs_now[indnow]) for indnow in indsnow])

        if self.lenbound is not None:
            maxlen = min(maxlen, self.lenbound)

        xnow = self.ignore * np.ones((maxlen, batchsize), dtype=np.long)
        lastinds = []

        for indoutnow, indnow in enumerate(indsnow):
            docnow = np.asarray(self.inds_docs_now[indnow], dtype=np.long)

            lennow = min(maxlen, docnow.shape[0])

            xnow[:lennow, indoutnow] = docnow[:lennow]
            lastinds.append(max(lennow-1,0))

        return xnow, self.Y[indsnow,:], np.asarray(lastinds, dtype=np.long)

    def get_batch(self, batchsize):

        if self.batchtype == 'iterated':
            if self.isfinished:
                self.reset_iter()
            return self.get_batch_iterated(batchsize)
        elif self.batchtype == 'random':
            return self.get_batch_random(batchsize)

    def get_batch_random(self, batchsize=None):

        if batchsize is None:
            batchsize = self.batchsize

        indsnow = np.random.randint(0, self.n_data, batchsize)

        return self.get_batch_byinds(indsnow)

    def get_batch_iterated(self, batchsize=None):

        if batchsize is None:
            batchsize = self.batchsize

        maxind = self.iternow + batchsize

        if maxind > self.n_data - 1:
            maxind = self.n_data - 1
            self.isfinished = True

        indsnow = [indnow for indnow in np.arange(self.iternow, maxind)]

        self.iternow += batchsize

        return self.get_batch_byinds(indsnow)


class MeanEmbeddingNetwork(nn.Module):
    def __init__(self, n_words, embdim, sizes, ignore_ind, outtype='sigmoid', lr=0.01, use_gpu=False):

        super(MeanEmbeddingNetwork, self).__init__()
        self.n_words = n_words
        self.embdim = embdim
        self.sizes = sizes
        self.ignore_ind = ignore_ind
        self.lr = lr
        self.use_gpu = use_gpu
        self.outtype = outtype

        self.outdim = sizes[-1]
        self.emb = nn.Embedding(n_words + 1, embdim, ignore_ind)

        self.n_layers = len(sizes)
        self.dims = [(embdim, sizes[0])]
        if len(sizes) > 1:
            for i in range(len(sizes) - 1):
                self.dims.append((sizes[i], sizes[i + 1]))

        self.layers = nn.ModuleList()
        for dim1, dim2 in self.dims:
            self.layers.append(nn.Linear(dim1, dim2))

        self.optimizer = None

        self.sigm = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.monitor = {'iters_train': [], 'iters_test': [], 'losses_train': [], 'losses_test': [],
                        'accs_train': [], 'accs_test': []}

        self.step_now = None
        self.lossnow = None
        self.countnow = None
        self.accnow = None

    def initialize(self):

        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        self.optimizer.zero_grad()

        self.step_now = -1
        self.lossnow = 0
        self.countnow = 0
        self.accnow = 0

        if self.use_gpu:
            self.cuda(0)

    def init_emb(self, embeddings):

        if self.use_gpu:
            self.emb.weight.data[:-1, :] = torch.from_numpy(np.asarray(embeddings, np.float32).copy()).cuda(0)
        else:
            self.emb.weight.data[:-1, :] = torch.from_numpy(np.asarray(embeddings, np.float32).copy())

    def gpu_move(self):

        self.cuda(0)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        self.use_gpu = True

    def cpu_move(self):

        self.cpu()

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cpu()

        self.use_gpu = False

    def forward(self, II):

        X = self.emb(II)

        mask = (X != 0).float()
        mask.requires_grad = False

        X = (X * mask).sum(dim=0) / torch.clamp(mask.sum(dim=0), min=1)

        for i in range(self.n_layers):

            X = self.layers[i](X)

            if i < self.n_layers - 1:
                X = F.relu(X)

        if self.outtype == 'linear':
            return X
        elif self.outtype == 'sigmoid':
            return F.sigmoid(X)
        elif self.outtype == 'softmax':
            return F.softmax(X)

    def find_acc(self, Yhat, Y):

        dummy, yhat = torch.max(Yhat, 1)
        dummy, y = torch.max(Y, 1)

        acc = (yhat == y).data

        if self.use_gpu:
            acc = acc.type(torch.cuda.FloatTensor)
        else:
            acc = acc.type(torch.FloatTensor)

        return acc.mean()

    def change_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def loss01(self, Yhat, Y):
        return F.mse_loss(Yhat, Y)

    def test_model(self, batcher_test):

        accnow_test = 0
        lossnow_test = 0
        countnow_test = 0

        batcher_test.reset_iter()

        tt1 = time.time()

        while not batcher_test.isfinished:

            Xnow, Ynow, lastinds = batcher_test.get_batch_iterated()
            sizenow = Xnow.shape[1]
            Xnow = Variable(torch.from_numpy(Xnow), requires_grad=False)
            Ynow = Variable(torch.from_numpy(Ynow), requires_grad=False)
            # lastinds = torch.from_numpy(lastinds)

            if self.use_gpu:
                Xnow = Xnow.cuda(0)
                Ynow = Ynow.cuda(0)
                # lastinds = lastinds.cuda(0)

            Yhat = self.forward(Xnow)
            loss = self.loss01(Yhat, Ynow)

            if self.use_gpu:
                loss = loss.cpu()

            lossnow_test += loss.data.numpy()[0] * sizenow
            accnow_test += self.find_acc(Yhat, Ynow) * sizenow
            countnow_test += sizenow

        loss_print = lossnow_test / countnow_test
        acc_print = accnow_test / countnow_test
        time_print = time.time() - tt1

        print('Iter: %i, time: %f, TEST Loss: %f, acc: %f' %
              (self.step_now+1, time_print, loss_print, acc_print), flush=True)

    def train_model(self, batcher_train, batcher_test=None, n_steps=10, update_step=1, fix_embeds=False,
                    disp_step=1, test_step=1, save_step=1, lr=0.01, savename=None):

        print("training..", flush=True)

        if fix_embeds:
            self.emb.weight.requires_grad = False
        else:
            self.emb.weight.requires_grad = True

        self.change_lr(lr)

        self.train()

        t1 = time.time()

        for self.step_now in range(self.step_now+1, self.step_now + n_steps + 1):

            Xnow, Ynow, lastinds = batcher_train.get_batch_random()
            Xnow = Variable(torch.from_numpy(Xnow), requires_grad=False, volatile=False)
            Ynow = Variable(torch.from_numpy(Ynow), requires_grad=False, volatile=False)
            # lastinds = torch.from_numpy(lastinds)

            if self.use_gpu:
                Xnow = Xnow.cuda(0)
                Ynow = Ynow.cuda(0)
                # lastinds = lastinds.cuda(0)

            Yhat = self.forward(Xnow)

            loss = self.loss01(Yhat, Ynow)

            loss.backward()

            if (self.step_now+1) % update_step == 0:
                print('Iter: %i, Updating parameters' % (self.step_now + 1), flush=True)
                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.use_gpu:
                loss = loss.cpu()

            self.lossnow += loss.data.numpy()[0]
            self.accnow += self.find_acc(Yhat, Ynow)
            self.countnow += 1

            del Xnow, Ynow

            if (self.step_now+1) % disp_step == 0:
                loss_print = self.lossnow / self.countnow
                acc_print = self.accnow / self.countnow

                time_print = time.time() - t1
                t1 = time.time()

                print('Iter: %i, time: %f, TRAIN Loss: %f, acc: %f' %
                      (self.step_now+1, time_print, loss_print, acc_print), flush=True)

                self.countnow = 0
                self.accnow = 0
                self.lossnow = 0

            if (self.step_now+1) % test_step == 0:
                self.test_model(batcher_test)

            if (self.step_now+1) % save_step == 0:
                if savename is not None:
                    torch.save(self, savename + '_step' + str(self.step_now+1))

