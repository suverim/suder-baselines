# onlineldavb.py: Package of functions for fitting Latent Dirichlet
# Allocation (LDA) with online variational Bayes (VB).
#
# Copyright (C) 2010  Matthew D. Hoffman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys, re, time, string
import numpy as n
from scipy.special import gammaln, psi

n.random.seed(100000001)
meanchangethresh = 0.001


def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return (psi(alpha) - psi(n.sum(alpha)))
    # n.sum(alpha, 1) sums the rows of alpha
    return (psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis])


class OnlineLDA:
    """
    Implements online VB for LDA as described in (Hoffman et al. 2010).
    """

    def __init__(self, vocab_dict, K, D, alpha, eta, tau0, kappa):
        """
        Arguments:
        K: Number of topics
        vocab: A set of words to recognize. When analyzing documents, any word
           not in this set will be ignored.
        D: Total number of documents in the population. For a fixed corpus,
           this is the size of the corpus. In the truly online setting, this
           can be an estimate of the maximum number of documents that
           could ever be seen.
        alpha: Hyperparameter for prior on weight vectors theta
        eta: Hyperparameter for prior on topics beta
        tau0: A (positive) learning parameter that downweights early iterations
        kappa: Learning rate: exponential decay rate---should be between
             (0.5, 1.0] to guarantee asymptotic convergence.
        Note that if you pass the same set of D documents in every time and
        set kappa=0 this class can also be used to do batch VB.
        """

        self._vocab = vocab_dict
        self._K = K
        self._W = len(self._vocab) + 1
        self._D = D
        self._alpha = alpha
        self._eta = eta
        self._tau0 = tau0 + 1
        self._kappa = kappa
        self._updatect = 0

        # Initialize the variational distribution q(beta|lambda)
        # K x W
        self._lambda = 1 * n.random.gamma(100., 1. / 100., (self._K, self._W))
        # K x W
        self._Elogbeta = dirichlet_expectation(self._lambda)
        # K x W
        self._expElogbeta = n.exp(self._Elogbeta)

    #   docs - a minibatch of documents

    def get_acc(self, docs_now, y_now, topic2class_now=None):

        outs_now = self.do_e_step(docs_now)

        gamma_now = outs_now[0]

        if topic2class_now is None:
            topic2class_now = get_class_mapper(y_now, gamma_now)

        yhat_now = topic2class_now[gamma_now.argmax(axis=1)]

        acc = (yhat_now == y_now).sum() / y_now.shape[0]

        return acc, topic2class_now

    def do_e_step(self, docs):
        """
        Given a mini-batch of documents, estimates the parameters
        gamma controlling the variational distribution over the topic
        weights for each document in the mini-batch.
        Arguments:
        docs:  List of D documents. Each document must be represented
               as a string. (Word order is unimportant.) Any
               words not in the vocabulary will be ignored.
        Returns a tuple containing the estimated values of gamma,
        as well as sufficient statistics needed to update lambda.
        """

        if type(docs[0][0]) == int:
            docs = [docs]

        batchD = len(docs)

        # Initialize the variational distribution q(theta|gamma) for
        # the mini-batch
        #   shape = batchD x K
        gamma = 1 * n.random.gamma(100., 1. / 100., (batchD, self._K))
        #   shape = batchD x K
        Elogtheta = dirichlet_expectation(gamma)
        # e ^ Elogtheta
        expElogtheta = n.exp(Elogtheta)

        #   sufficient statistics
        #   shape = K x vocab_size
        sstats = n.zeros(self._lambda.shape)
        # Now, for each document d update that document's gamma and phi
        it = 0
        meanchange = 0

        for d in range(0, batchD):
            # get the word ids for every word in the document
            ids = docs[d][0]

            # get the word counts
            cts = docs[d][1]

            # get the row of gamma for this document
            gammad = gamma[d, :]

            # get the E[log(theta)] row for this document
            Elogthetad = Elogtheta[d, :]

            # get the exponential terms for the phi update

            expElogthetad = expElogtheta[d, :]
            expElogbetad = self._expElogbeta[:, ids]

            # The optimal phi_{dwk} is proportional to
            # expElogthetad_k * expElogbetad_w. phinorm is the normalizer.
            # phinorm is size of second dimension of expElogbetad
            #    (number of words in the batch)
            phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100

            # Iterate between gamma and phi until convergence
            for it in range(0, 100):
                lastgamma = gammad

                # We represent phi implicitly to save memory and time.
                # Substituting the value of the optimal phi back into
                # the update for gamma gives this update. Cf. Lee&Seung 2001.
                gammad = self._alpha + expElogthetad * \
                                       n.dot(cts / phinorm, expElogbetad.T)

                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = n.exp(Elogthetad)

                phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100
                # If gamma hasn't changed much, we're done.
                meanchange = n.mean(abs(gammad - lastgamma))
                if (meanchange < meanchangethresh):
                    break
            gamma[d, :] = gammad

            # Contribution of document d to the expected sufficient
            # statistics for the M step.
            # Adds contributions to the word ids found in the doc.
            # n.outer is the outer product
            # expElogthetad.T is size K
            # cts/phinorm is size {num_words_in_doc}
            sstats[:, ids] += n.outer(expElogthetad.T, cts / phinorm)

        # This step finishes computing the sufficient statistics for the
        # M step, so that
        # sstats[k, w] = \sum_d n_{dw} * phi_{dwk}
        # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
        sstats = sstats * self._expElogbeta

        return ((gamma, sstats))

    def update_lambda(self, docs):
        """
        First does an E step on the mini-batch given in wordids and
        wordcts, then uses the result of that E step to update the
        variational parameter matrix lambda.
        Arguments:
        docs:  List of D documents. Each document must be represented
               as a string. (Word order is unimportant.) Any
               words not in the vocabulary will be ignored.
        Returns gamma, the parameters to the variational distribution
        over the topic weights theta for the documents analyzed in this
        update.
        Also returns an estimate of the variational bound for the
        entire corpus for the OLD setting of lambda based on the
        documents passed in. This can be used as a (possibly very
        noisy) estimate of held-out likelihood.
        """

        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        # p_t = (t_0 + t)^{-k}
        # _updatect tracks the number of batches analyzed
        rhot = pow(self._tau0 + self._updatect, -self._kappa)
        self._rhot = rhot

        # Do an E step to update gamma, phi | lambda for this
        # mini-batch. This also returns the information about phi that
        # we need to update lambda.
        (gamma, sstats) = self.do_e_step(docs)

        # Estimate held-out likelihood for current values of lambda.
        bound = self.approx_bound(docs, gamma)

        # Update lambda based on documents.
        # Lambda update from the paper.
        if len(docs) != 0:
            self._lambda = self._lambda * (1 - rhot) + \
                           rhot * (self._eta + self._D * sstats / len(docs))

        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)
        self._updatect += 1

        return (gamma, bound)

    def approx_bound(self, docs, gamma):
        """
        Estimates the variational bound over *all documents* using only
        the documents passed in as "docs." gamma is the set of parameters
        to the variational distribution q(theta) corresponding to the
        set of documents passed in.
        The output of this function is going to be noisy, but can be
        useful for assessing convergence.
        """
        if type(docs[0][0]) == int:
            docs = [docs]

        batchD = len(docs)

        score = 0
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = n.exp(Elogtheta)

        # E[log p(docs | theta, beta)]
        for d in range(0, batchD):
            gammad = gamma[d, :]
            ids = docs[d][0]
            cts = n.array(docs[d][1])
            phinorm = n.zeros(len(ids))
            for i in range(0, len(ids)):
                temp = Elogtheta[d, :] + self._Elogbeta[:, ids[i]]
                tmax = max(temp)
                phinorm[i] = n.log(sum(n.exp(temp - tmax))) + tmax
            score += n.sum(cts * phinorm)
            oldphinorm = phinorm
            phinorm = n.dot(expElogtheta[d, :], self._expElogbeta[:, ids])
            score += n.sum(cts * n.log(phinorm))

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += n.sum((self._alpha - gamma) * Elogtheta)
        score += n.sum(gammaln(gamma) - gammaln(self._alpha))
        score += sum(gammaln(self._alpha * self._K) - gammaln(n.sum(gamma, 1)))

        # Compensate for the subsampling of the population of documents
        if len(docs) != 0:
            score = score * self._D / len(docs)

        # E[log p(beta | eta) - log q (beta | lambda)]
        score = score + n.sum((self._eta - self._lambda) * self._Elogbeta)
        score = score + n.sum(gammaln(self._lambda) - gammaln(self._eta))
        score = score + n.sum(gammaln(self._eta * self._W) -
                              gammaln(n.sum(self._lambda, 1)))

        return (score)


class Batcher:
    def __init__(self, df_train, df_test, textdict, batchsize,
                 preprocessor, tokenizer, indexer, vocab_dict, idcolname='TextId'):

        self.df_train = df_train
        self.df_test = df_test
        self.textdict = textdict
        self.batchsize = batchsize

        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.indexer = indexer
        self.vocab_dict = vocab_dict
        self.idcolname = idcolname

        self.n_docs_train = df_train.shape[0]

        self.inds_train = n.array([x for x in range(0, df_train.shape[0])])
        self.count_train = 0

        self.docs_train = None
        self.docs_test = None

    def fill_train(self):

        self.docs_train = []

        for ind, row in self.df_train.iterrows():
            textnow = self.preprocessor(self.textdict[row[self.idcolname]])
            wordsnow = self.tokenizer(textnow)
            indsnow = self.indexer(wordsnow, self.vocab_dict)

            self.docs_train.append(indsnow)

    def fill_test(self):

        self.docs_test = []

        for ind, row in self.df_test.iterrows():
            textnow = self.preprocessor(self.textdict[row[self.idcolname]])
            wordsnow = self.tokenizer(textnow)
            indsnow = self.indexer(wordsnow, self.vocab_dict)

            self.docs_test.append(indsnow)

    def get_train_batch(self):

        indstmp = n.arange(self.count_train, self.count_train + self.batchsize) % self.df_train.shape[0]

        dfnow = self.df_train.iloc[self.inds_train[indstmp]]

        self.count_train += self.batchsize

        out = []

        for ind, row in dfnow.iterrows():
            textnow = self.preprocessor(self.textdict[row[self.idcolname]])
            wordsnow = self.tokenizer(textnow)
            indsnow = self.indexer(wordsnow, self.vocab_dict)

            out.append(indsnow)

        return out


def get_class_mapper(y, gamma):

    n_class = n.unique(y).shape[0]

    n_topics = gamma.shape[1]

    scores_class = n.zeros((n_class, n_topics))

    for cl in range(n_class):

        scores_class[cl,:] = gamma[y == cl,:].mean(axis=0)

    return n.argmax(scores_class, axis=0)