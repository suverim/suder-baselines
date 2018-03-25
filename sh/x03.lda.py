#!/usr/bin/python3

import pandas as pd
import sys
from subprocess import call
import pickle
import datetime as dt
import json
import os
import time

sys.path.append('../bin')
import utils04 as utils
import onlineldavb02 as LDA02

expid = sys.argv[1]
collection = sys.argv[2]
n_topics = int(sys.argv[3])
vocsize = int(sys.argv[4])

print("expid: %s\ncollection: %s\nnum of topics: %i\nVocabulary size: %i\n" % (expid, collection, n_topics, vocsize), flush=True)
if len(sys.argv) == 6:
    print(sys.argv[5])
elif len(sys.argv) > 6:
    raise Exception("Number of arguments is greater than 5")

# File inputs, outputs

datafolder = os.path.join('../data/', collection)

metafile = os.path.join(datafolder, 'meta_'+collection+'.csv')
jsonfile = os.path.join(datafolder, 'texts_'+collection+'.json')
vocabfile = os.path.join(datafolder, 'vocab02_longroot.pickle')
mapperfile = os.path.join(datafolder, 'vocab01_root_mapper01.txt')

outfolder = os.path.join(datafolder, 'lda'+expid)
outmodelfolder = os.path.join(outfolder, 'models')
out_lda = os.path.join(outfolder, 'lda.pickle')
out_vocab_text = os.path.join(outfolder, 'vocabulary.txt')
out_vocab_pickle = os.path.join(outfolder, 'vocabulary.pickle')

call(['mkdir', '-p', outmodelfolder])

# load word map file

word_mapper = {}
with open(mapperfile) as f:
    for line in f:
        key, val = line.strip().split(' ')
        word_mapper[key] = val


# Parameters

batchsize = 100
nepochs = 6
save_every = 1000


stop_words=utils.TURKISH_STOP_WORDS
idcolname = 'TextId'


print("Loading datasets", flush=True)

df_meta = pd.read_csv(metafile, parse_dates=['Datetime', 'Date'], infer_datetime_format=True)


df_train = df_meta[df_meta.Datetime < dt.datetime(2016,9,1)]
df_test = df_meta[df_meta.Datetime >= dt.datetime(2016,9,1)]

# df_train = df_train.iloc[:100]
# df_test = df_test.iloc[:100]

with open(jsonfile) as f:
    texts = json.load(f)

texts = {int(idnow): valnow for idnow, valnow in texts.items()}

with open(vocabfile, 'rb') as f:
    vocabulary = pickle.load(f)

print("Loaded data", flush=True)

_ = vocabulary.reduce_voc(n_words=vocsize, n_min_count=None, stop_words=stop_words, rm_startswith='_S_')

print("Reduced vocabulary to %i words" % (len(vocabulary.vocab_reduced)), flush=True)

vocabulary.write_voc_reduced(out_vocab_text)

with open(out_vocab_pickle, 'wb') as f:
    print("Pickling reduced vocabulary to", out_vocab_pickle, flush=True)
    pickle.dump(vocabulary.vocab_reduced, f)

vocab_dict = vocabulary.get_vocab_reduced_dict()


labelnames = [namenow for namenow in df_train.Category.unique()]

label_mapper = {name: ind for ind, name in enumerate(labelnames)}

y_train = df_train.Category.apply(lambda x: label_mapper[x]).values
y_test = df_test.Category.apply(lambda x: label_mapper[x]).values


def tokenizer(textnow): return utils.tokenizer02(textnow, mapper=word_mapper, vocabulary=vocab_dict)
preprocessor = utils.preprocess_tr
indexer = utils.indexer01

print("Creating LDA, filling documents", flush=True)
batcher = LDA02.Batcher(df_train, df_test, texts, batchsize, preprocessor, tokenizer, indexer, vocab_dict)
batcher.fill_test()
batcher.fill_train()
print("Finished getting documents", flush=True)


D = df_train.shape[0]
n_batches = int(D/batchsize)
W = len(vocabulary.vocab_reduced)

# Initialize the algorithm with alpha=1/K, eta=1/K, tau_0=1024, kappa=0.7
alpha = 1./n_topics  # prior on topic weights theta
eta = 1./n_topics  # prior on p(w|topic) Beta
tau_0 = 1024 # learning parameter to downweight early documents
kappa = 0.7 # learning parameter; decay factor for influence of batches

niters = n_batches * nepochs
log_every = int(n_batches/3)

lda = LDA02.OnlineLDA(vocabulary.vocab_reduced, n_topics, D, alpha, 1./n_topics, tau_0, kappa)

t1 = time.time()

for iternow in range(0, niters):

    if iternow % 100 == 0:
        print(iternow, end=', ', flush=True)

    docset = batcher.get_train_batch()

    (gamma, bound) = lda.update_lambda(docset)

    if iternow % log_every == 0:
        acc_train, topic2class_now = lda.get_acc(batcher.docs_train, y_train)
        acc_test, _ = lda.get_acc(batcher.docs_test, y_test, topic2class_now=topic2class_now)

        print("\nIter %i, epoch: %i - %.2f%%, train-acc: %f, test-acc: %f, time: %f\n" %
              (iternow, iternow // n_batches, round((iternow % n_batches) / n_batches * 100, 2),
               acc_train, acc_test, time.time() - t1), flush=True)

        t1 = time.time()

    if iternow % save_every == 0 or iternow == n_batches:
        with open(os.path.join(outmodelfolder, "lda_iter" + str(iternow).zfill(5)+'.pickle'), 'wb') as f:
            pickle.dump(lda, f)

print("Finished training", flush=True)

acc_train, topic2class_now = lda.get_acc(batcher.docs_train, y_train)
acc_test, _ = lda.get_acc(batcher.docs_test, y_test, topic2class_now=topic2class_now)

with open(os.path.join(outfolder, 'accuracy.txt'), 'w') as f:
    f.write("Train: %f" % (acc_train) + '\n')
    f.write("Test: %f" % (acc_test) + '\n')

print("Train-acc: %f, test-acc: %f, time: %f" %(acc_train, acc_test, time.time() - t1), flush=True)

with open(out_lda, 'wb') as f:
    pickle.dump(lda, f)

