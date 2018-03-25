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
import logging

from subprocess import call
from gensim.models import Word2Vec

sys.path.append('../bin')

import utils04 as utils

collection = sys.argv[1]
featid = sys.argv[2]
dim = int(sys.argv[3])
n_epochs = int(sys.argv[4])
n_workers = int(sys.argv[5])

folder_collection = os.path.join('../data', collection)
outfolder = os.path.join(folder_collection, 'we'+featid)

metafile = os.path.join(folder_collection, 'meta_'+collection+'.csv')
jsonfile = os.path.join(folder_collection, 'texts_'+collection+'.json')
mapperfile = os.path.join(folder_collection, 'vocab01_root_mapper01.txt')

outfile = os.path.join(outfolder, 'we.pickle')
out_vocab_text = os.path.join(outfolder, 'vocabulary.txt')
out_vectors = os.path.join(outfolder, 'we_vectors.pickle')

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)

# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
#                     datefmt='%m-%d %H:%M',
#                     filename='/temp/myapp.log',
#                     filemode='w')


call(['mkdir', '-p', outfolder])

logging.info("\n\ncollection: %s\nfeatid: %s\ndim: %i\nnum of epochs: %i, num_workers: %i" %
             (collection, featid, dim, n_epochs, n_workers))
if len(sys.argv) == 7:
    logging.info("\n"+sys.argv[6])
elif len(sys.argv) > 7:
    raise Exception("Number of arguments is greater than 6")


df_meta = pd.read_csv(metafile, parse_dates=['Datetime', 'Date'], infer_datetime_format=True)
df_train = df_meta[df_meta.Datetime < dt.datetime(2016,9,1)]
df_test = df_meta[df_meta.Datetime >= dt.datetime(2016,9,1)]

with open(jsonfile) as f:
    texts = json.load(f)
texts = {int(idnow): valnow for idnow, valnow in texts.items()}

mapper = {}
with open(mapperfile) as f:
    for line in f:
        word1, word2 = line.strip().split()
        mapper[word1] = word2

logging.info("Loaded data")


def tokenizer(textnow): return utils.tokenizer02(textnow, mapper=mapper, vocabulary=None)
preprocessor = utils.preprocess_tr


def df2docs(dfnow):
    docsnow = []

    for ind, row in dfnow.iterrows():
        textnow = preprocessor(texts[row['TextId']])
        wordsnow = tokenizer(textnow)

        docsnow.append(wordsnow)

    return docsnow

logging.info("Processing and tokenizing train data")
docs_train = df2docs(df_train)
logging.info("Done!")

model = Word2Vec(size=dim, window=20, negative=5, sample=0, hs=0, iter=n_epochs, compute_loss=True, workers=n_workers)


logging.info("Building vocabulary")
model.build_vocab(docs_train)

with open(out_vocab_text, 'w') as f:
    for word in model.wv.index2word:
        f.write(word+'\n')


logging.info("Starting Training")
model.train(docs_train, total_examples=len(docs_train), epochs=model.iter)


model.save(outfile)

with open(out_vectors, 'wb') as f:
    pickle.dump(model.wv.vectors, f)
    pickle.dump(model.wv.index2word, f)

