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

sys.path.append('../bin')
sys.path.append('.')

import utils04


featid = sys.argv[1]
expid = sys.argv[2]
collection = sys.argv[3]
C = float(sys.argv[4])

print("Collection: %s\nFeature ID: %s\nExperiment ID: %s\nC: %f" % (collection, featid, expid, C), flush=True)

if len(sys.argv) == 6:
    print("\n"+sys.argv[5])
elif len(sys.argv) > 6:
    raise Exception("Number of arguments is greater than 5")

folder_collection = os.path.join('../data', collection)
folder_feature = os.path.join(folder_collection, 'tfidf'+featid)
folder_exp = os.path.join(folder_feature, 'exp'+expid)

metafile = os.path.join(folder_collection, 'meta_'+collection+'.csv')
tfidffile = os.path.join(folder_feature, 'tfidf.pickle')
outmodelfile = os.path.join(folder_exp, 'model.pickle')
outinfofile = os.path.join(folder_exp, 'accuracy.txt')

call(['mkdir', '-p', folder_exp])

def tokenizer(textnow): return None

print("Loading datasets:\n%s\n%s\n" % (metafile, tfidffile), flush=True)

df_meta = pd.read_csv(metafile, parse_dates=['Datetime', 'Date'], infer_datetime_format=True)


df_train = df_meta[df_meta.Datetime < dt.datetime(2016,9,1)]
df_test = df_meta[df_meta.Datetime >= dt.datetime(2016,9,1)]

with open(tfidffile, 'rb') as f:
    vectorizer = pickle.load(f)
    tfidf_train = pickle.load(f)
    tfidf_test = pickle.load(f)

print("Loaded datasets", flush=True)


labelnames = [namenow for namenow in df_train.Category.unique()]

label_mapper = {name: ind for ind, name in enumerate(labelnames)}

y_train = df_train.Category.apply(lambda x: label_mapper[x]).values
y_test = df_test.Category.apply(lambda x: label_mapper[x]).values

svm01 = svm.LinearSVC(C=C, verbose=100, dual=False)

t1 = time.time()
svm01 = svm01.fit(tfidf_train, y_train)
print("Finished training, time:", time.time() - t1, flush=True)

yhat_test = svm01.predict(tfidf_test)
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
