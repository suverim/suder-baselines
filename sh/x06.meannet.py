#!/usr/bin/python3

import sys
import importlib
import pandas as pd
import json
import datetime as dt
import numpy as np
import time
import pickle
import os
import argparse
from subprocess import call

sys.path.append('../bin')

import meannetwork as mn
import utils04 as utils

parser = argparse.ArgumentParser()

parser.add_argument("collection", help="collection to use (sabah)", type=str)
parser.add_argument("featid", help="Embedding Id (we02)", type=str)
parser.add_argument("expid", help="Experiment Id (exp01)", type=str)

parser.add_argument("-bs", "--batchsize", default=100, type=int)
parser.add_argument("-s", "--sizes", default="100")
parser.add_argument("-bt", "--batchtype", default='random')
parser.add_argument("-lb", "--lenbound", default=1000, type=int)
parser.add_argument("-ot", "--outtype", default='sigmoid')
parser.add_argument("-lr", "--learningrate", default="0.1", type=float)
parser.add_argument("--gpu", action="store_true")
parser.add_argument("-e", "--n_epochs", default=10, type=int)
parser.add_argument("--update_step", default=1, type=int)
parser.add_argument("--disp_step", default=0, type=float)
parser.add_argument("--test_step", default=0, type=float)
parser.add_argument("--save_step", default=0, type=float)

args = parser.parse_args()

collection = args.collection
featid = args.featid
expid = args.expid

batchsize = args.batchsize
sizes = [int(sizenow) for sizenow in args.sizes.split(",")]
batchtype = args.batchtype
lenbound = args.lenbound
outtype = args.outtype
learningrate = args.learningrate
usegpu = args.gpu
n_epochs = args.n_epochs
update_step = args.update_step
disp_step = args.disp_step
test_step = args.test_step
save_step = args.save_step

folder_collection = os.path.join('../data', collection)
folder_feature = os.path.join(folder_collection, 'we'+featid)
folder_exp = os.path.join(folder_feature, 'exp'+expid)
folder_outmodels = os.path.join(folder_exp, 'models')

mapperfile = os.path.join(folder_collection, 'vocab01_root_mapper01.txt')
metafile = os.path.join(folder_collection, 'meta_'+collection+'.csv')
jsonfile = os.path.join(folder_collection, 'texts_'+collection+'.json')
we_vectors_file = os.path.join(folder_feature, 'we_vectors.pickle')
ind_docs_file = os.path.join(folder_feature, 'ind_docs.pickle')

outmodelfile_base = os.path.join(folder_outmodels, 'model')
outinfofile = os.path.join(folder_exp, 'info.txt')

call(['mkdir', '-p', folder_outmodels])

with open(outinfofile, 'a') as f:
    f.write('\n')
    for arg in sorted(vars(args)):
        print(arg, getattr(args, arg), flush=True)
        f.write(arg + ': ' + str(getattr(args, arg)) + '\n')
    f.write('\n')

df_meta = pd.read_csv(metafile, parse_dates=['Datetime', 'Date'], infer_datetime_format=True)
df_train = df_meta[df_meta.Datetime < dt.datetime(2016,9,1)]
df_test = df_meta[df_meta.Datetime >= dt.datetime(2016,9,1)]

with open(we_vectors_file, 'rb') as f:
    embeddings = pickle.load(f)
    emb_vocab_list = pickle.load(f)

emb_vocab_w2i = {wordnow: keynow for keynow, wordnow in enumerate(emb_vocab_list)}

labelnames = [namenow for namenow in df_train.Category.unique()]
label_mapper = {name: ind for ind, name in enumerate(labelnames)}
y_train = df_train.Category.apply(lambda x: label_mapper[x]).values
y_test = df_test.Category.apply(lambda x: label_mapper[x]).values

if os.path.isfile(ind_docs_file):
    with open(ind_docs_file, 'rb') as f:
        ind_docs_train = pickle.load(f)
        ind_docs_test = pickle.load(f)
else:

    with open(jsonfile) as f:
        texts = json.load(f)
    texts = {int(idnow): valnow for idnow, valnow in texts.items()}

    mapper = {}
    with open(mapperfile) as f:
        for line in f:
            word1, word2 = line.strip().split()
            mapper[word1] = word2

    def tokenizer(textnow): return utils.tokenizer02(textnow, mapper=mapper, vocabulary=emb_vocab_w2i)
    preprocessor = utils.preprocess_tr

    def df2inds(dfnow):
        ind_docs_now = []

        for ind, row in dfnow.iterrows():
            textnow = preprocessor(texts[row['TextId']])
            wordsnow = tokenizer(textnow)
            indsnow = [emb_vocab_w2i[wordnow] for wordnow in wordsnow]

            ind_docs_now.append(indsnow)

        return ind_docs_now

    ind_docs_train = df2inds(df_train)
    ind_docs_test = df2inds(df_test)

    with open(ind_docs_file, 'wb') as f:
        pickle.dump(ind_docs_train, f)
        pickle.dump(ind_docs_test, f)


batcher_train = mn.Batcher(ind_docs_train, y_train, batchsize=batchsize, batchtype=batchtype, ignore=None, lenbound=lenbound)
batcher_test = mn.Batcher(ind_docs_test, y_test, batchsize=batchsize, ignore=batcher_train.ignore, lenbound=lenbound)

sizes.append(int(y_train.max()+1))

nn01 = mn.MeanEmbeddingNetwork(embeddings.shape[0],
                               embeddings.shape[1],
                               sizes,
                               int(batcher_train.ignore),
                               outtype=outtype, lr=learningrate, use_gpu=False)

nn01.initialize()

nn01.init_emb(embeddings)

if usegpu:
    nn01.gpu_move()

n_data = df_train.shape[0] # 361425
n_steps_eachbatch = n_data // batchsize #3614
n_steps = n_steps_eachbatch * n_epochs #36140

if update_step == 0:
    update_step = n_steps_eachbatch #3614

if disp_step == 0:
    disp_step = n_steps_eachbatch
elif disp_step < 1:
    disp_step = int(n_steps_eachbatch * disp_step) #1807
else:
    disp_step = int(disp_step)

if test_step == 0:
    test_step = update_step #3614
elif test_step < 1:
    test_step = int(n_steps_eachbatch * test_step)
else:
    test_step = int(test_step)

if save_step == 0:
    save_step = update_step
elif save_step < 1:
    save_step = int(n_steps_eachbatch * save_step)

nn01.train_model(batcher_train, batcher_test, n_steps=n_steps, update_step=update_step, fix_embeds=False,
                 disp_step=disp_step, test_step=test_step, save_step=save_step, savename=outmodelfile_base)