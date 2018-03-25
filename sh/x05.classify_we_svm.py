#!/usr/bin/python3

import os
import sys
import pandas as pd
import datetime as dt
import pickle
from sklearn import svm
import sklearn
from subprocess import call
import time
import json
import numpy as np


sys.path.append('../bin')
sys.path.append('.')

import utils04 as utils

collection = sys.argv[1]
featid = sys.argv[2]
expid = sys.argv[3]

C = float(sys.argv[4])

print("Collection: %s\nFeature ID: %s\nExperiment ID: %s\nC: %f" % (collection, featid, expid, C), flush=True)

if len(sys.argv) == 6:
    print("\n"+sys.argv[5], flush=True)
elif len(sys.argv) > 6:
    raise Exception("Number of arguments is greater than 5")

folder_collection = os.path.join('../data', collection)
folder_feature = os.path.join(folder_collection, 'we'+featid)
folder_exp = os.path.join(folder_feature, 'exp'+expid)

mapperfile = os.path.join(folder_collection, 'vocab01_root_mapper01.txt')
metafile = os.path.join(folder_collection, 'meta_'+collection+'.csv')
jsonfile = os.path.join(folder_collection, 'texts_'+collection+'.json')
we_vectors_file = os.path.join(folder_feature, 'we_vectors.pickle')

outmodelfile = os.path.join(folder_exp, 'model.pickle')
outinfofile = os.path.join(folder_exp, 'accuracy.txt')

call(['mkdir', '-p', folder_exp])

print("Loading datasets:\n%s\n%s\n" % (metafile, we_vectors_file), flush=True)

df_meta = pd.read_csv(metafile, parse_dates=['Datetime', 'Date'], infer_datetime_format=True)


df_train = df_meta[df_meta.Datetime < dt.datetime(2016,9,1)]
df_test = df_meta[df_meta.Datetime >= dt.datetime(2016,9,1)]

# df_train = df_train.iloc[:10000]

with open(jsonfile) as f:
    texts = json.load(f)
texts = {int(idnow): valnow for idnow, valnow in texts.items()}

with open(we_vectors_file, 'rb') as f:
    embeddings = pickle.load(f)
    emb_vocab_list = pickle.load(f)

emb_vocab_w2i = {wordnow: keynow for keynow, wordnow in enumerate(emb_vocab_list)}

mapper = {}
with open(mapperfile) as f:
    for line in f:
        word1, word2 = line.strip().split()
        mapper[word1] = word2

print("Loaded datasets", flush=True)


def tokenizer(textnow): return utils.tokenizer02(textnow, mapper=mapper, vocabulary=None)
preprocessor = utils.preprocess_tr


featsize = embeddings.shape[1]


def df2emb(dfnow):

    emb_docs_now = np.zeros((dfnow.shape[0], featsize))
    count = 0

    for ind, row in dfnow.iterrows():
        textnow = preprocessor(texts[row['TextId']])
        wordsnow = tokenizer(textnow)

        embnow = np.zeros((featsize, ))

        count_word = 0

        for wordnow in wordsnow:
            if wordnow in emb_vocab_w2i:
                embnow += embeddings[emb_vocab_w2i[wordnow],:]
                count_word += 1

        if count_word == 0:
            count_word = 1

        emb_docs_now[count,:] = embnow / count_word

        count += 1

    return emb_docs_now

t1 = time.time()
print("Getting mean word embeddings for each document", end='... ', flush=True)
embs_train = df2emb(df_train)
print("Training finished, for test...", end=' ', flush=True)
embs_test = df2emb(df_test)
print("Done, time: %f" % (time.time() - t1), flush=True)

labelnames = [namenow for namenow in df_train.Category.unique()]
label_mapper = {name: ind for ind, name in enumerate(labelnames)}
y_train = df_train.Category.apply(lambda x: label_mapper[x]).values
y_test = df_test.Category.apply(lambda x: label_mapper[x]).values

# y_train = y_train[:10000]

svm01 = svm.LinearSVC(C=C, verbose=100, dual=False)

t1 = time.time()
svm01 = svm01.fit(embs_train, y_train)
print("Finished training, time:", time.time() - t1, flush=True)

yhat_test = svm01.predict(embs_test)
acc = (yhat_test == y_test).sum() / y_test.shape[0]

C = sklearn.metrics.confusion_matrix(y_test, yhat_test, sample_weight=None)

print("accuracy is %f" % (acc), flush=True)

print("Confusion matrix:\n", flush=True)

print(C)

with open(outmodelfile,'wb') as f:
    pickle.dump(svm01, f)

with open(outinfofile, 'w') as f:
    f.write(str(acc)+'\n\n')
    f.write(str(C)+'\n')
