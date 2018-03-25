#!/usr/bin/python3

import pandas as pd
import sys
from subprocess import call
import pickle
import datetime as dt
import json
import sklearn.feature_extraction.text as fe

sys.path.append('../bin')
import utils04 as utils


# File inputs, outputs

metafile = '../data/cumhuriyet/meta_cumhuriyet.csv'
jsonfile = '../data/cumhuriyet/texts_cumhuriyet.json'

outvocabtextfile = '../data/cumhuriyet/vocab01.txt'
outvocabpicklefile = '../data/cumhuriyet/vocab01.pickle'


# Parameters

preprocessor = utils.preprocess_tr
def tokenizer(textnow): return utils.tokenizer01(textnow, mapper=None)
idcolname = 'TextId'


print("Loading datasets", flush=True)

df_meta = pd.read_csv(metafile, parse_dates=['Datetime'], infer_datetime_format=True)


df_train = df_meta[df_meta.Datetime < dt.datetime(2016,9,1)]
df_test = df_meta[df_meta.Datetime >= dt.datetime(2016,9,1)]

# df_train = df_train.iloc[:100]
# df_test = df_test.iloc[:100]

with open(jsonfile) as f:
    texts = json.load(f)

texts = {int(idnow): valnow for idnow, valnow in texts.items()}

print("Loaded datasets\nCreating Vocabulary", flush=True)


train_iterator = utils.iter_texts(df_train, textdict=texts, idcolname=idcolname)


vocabulary = utils.Vocabulary(train_iterator, tokenizer=tokenizer, preprocessor=preprocessor)

print("Created vocabulary, Number of words: %i" % (len(vocabulary.word_counts)), flush=True)

vocabulary.write_voc_all(outvocabtextfile)

with open(outvocabpicklefile, 'wb') as f:
    print("Pickling vocab to", outvocabpicklefile, flush=True)
    pickle.dump(vocabulary, f)
