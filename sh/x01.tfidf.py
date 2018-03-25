#!/usr/bin/python3

import pandas as pd
import sys
from subprocess import call
import pickle
import datetime as dt
import json
import sklearn.feature_extraction.text as fe
import os

sys.path.append('../bin')
import utils04 as utils

expid = sys.argv[1]
collection = sys.argv[2]
vocsize = int(sys.argv[3])

print("expid: %s\ncollection: %s\nvocabulary size: %i" % (expid, collection, vocsize), flush=True)
if len(sys.argv) == 5:
    print("\n"+sys.argv[4])
elif len(sys.argv) > 5:
    raise Exception("Number of arguments is greater than 4")

# File inputs, outputs

metafile = os.path.join('../data/', collection, 'meta_'+collection+'.csv')
jsonfile = os.path.join('../data/', collection, 'texts_'+collection+'.json')
vocabfile = os.path.join('../data/', collection, 'vocab02_longroot.pickle')
mapperfile = os.path.join('../data/', collection, 'vocab01_root_mapper01.txt')

outfolder = os.path.join('../data', collection, 'tfidf'+expid)
out_tfidf = os.path.join(outfolder, 'tfidf.pickle')
out_vocab_text = os.path.join(outfolder, 'vocabulary.txt')
out_vocab_pickle = os.path.join(outfolder, 'vocabulary.pickle')


# load word map file

word_mapper = {}
with open(mapperfile) as f:
    for line in f:
        key, val = line.strip().split(' ')
        word_mapper[key] = val


# Parameters

def tokenizer(textnow): return utils.tokenizer01(textnow, mapper=word_mapper)
preprocessor = utils.preprocess_tr
stop_words=utils.TURKISH_STOP_WORDS
idcolname = 'TextId'


call(['mkdir', '-p', outfolder])

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

vocnow = vocabulary.get_vocab_reduced_dict()

vectorizer = fe.TfidfVectorizer(input='content', lowercase=False, preprocessor=preprocessor,
                                token_pattern="", tokenizer=tokenizer, stop_words=None,
                                vocabulary=vocabulary.get_vocab_reduced_dict(), norm='l1')

print("Created tf-idf vectorizer\nFitting to the training data", flush=True)

tfidf_train = vectorizer.fit_transform([texts[idnow] for idnow in df_train[idcolname].values])

print("Created tf-idf training vectors\nCreating test vectors", flush=True)

tfidf_test = vectorizer.transform([texts[idnow] for idnow in df_test[idcolname].values], copy=True)

print("Created tf-idf test vectors", flush=True)

with open(out_tfidf, 'wb') as f:
    print("Pickling data to", out_tfidf, flush=True)
    pickle.dump(vectorizer, f)
    pickle.dump(tfidf_train, f)
    pickle.dump(tfidf_test, f)

