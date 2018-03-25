# SuDer-v1-baselines

Baseline models for [suder corpus](https://github.com/suverim/suder-v1.0)

Corresponding paper: Document Classification of SuDer Turkish News Corpuses

## Required
Python-3, pytorch, numpy, pandas, sklearn, 

For Morphology, [zemberek](https://github.com/ahmetaa/zemberek-nlp) is used (May need to recompile the Java project).

For LDA, modified the code from https://github.com/blei-lab/onlineldavb

## Running

For each corpus (cumhuriyet or sabah), move the csv and json files to ./data/{corpus}/

Run scripts v01, v02, v03 and v04 for creating vocabularies and using zemberek to obtain word mapper.

### TF-IDF

Obtaining TF-IDF vectors:
```
./x01.tfidf.py 01 {corpus} {vocabulary size}
```
Using SVM to classify:
```
./x02.classify_tfidf_svm.py 01 01 {corpus} {C parameter}
```

### Word2vec

Obtaining word embeddings:
```
./x04.word2vec.py {corpus} 01 {embedding dimension} {#epochs} {#workers}
```
Using SVM to classify
```
./x05.classify_we_svm.py {corpus} 01 01 {C parameter}
```

### NN with Word Embeddings

Averages word embeddings within a document and apply Feed Forward Neural Network
```
./x06.meannet.py {corpus} 01 02 --learningrate 0.1 -e {#epochs} --sizes 50,50 --update_step 1 --disp_step 100 --test_step 0.5 --save_step 0.5 --gpu
```

### LDA
```
./x03.lda.py 01 {corpus} {#topics} {vocabulary size}
```
