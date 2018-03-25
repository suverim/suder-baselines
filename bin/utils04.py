# -*- coding: utf-8 -*-

import re
import string
import os
import numpy as np

from collections import Counter

TURKISH_STOP_WORDS = set()

for namenow in ['ahmetax', 'crodas', 'iso']:

    with open(os.path.dirname(os.path.abspath(__file__)) + '/tr_stopwords_' + namenow + '.txt') as f:

        for line in f:
            if line[0] != '#':
                line = line.strip().replace('İ', 'i')
                line = line.replace('I', 'ı')
                TURKISH_STOP_WORDS.add(line.lower())

TURKISH_STOP_WORDS = sorted(list(TURKISH_STOP_WORDS))

token_pattern_compiled = re.compile('(?u)\\b\\w\\w+\\b')

def preprocess_en(textnow):

    textnow = textnow.lower()

    textnow = textnow.replace('_', ' ')

    textnow = re.sub(r"([\d]+)", r" _N ", textnow)

    textnow = re.sub(r"([\w]{1})[\'\`\´\‘\’]([\w]+)", r"\1 _S_\2", textnow)

    return textnow


def preprocess_tr(textnow):

    textnow = textnow.replace('İ', 'i')
    textnow = textnow.replace('I', 'ı')
    textnow = textnow.lower()

    textnow = textnow.replace('_', ' ')

    textnow = re.sub(r"([\d]+)", r" _N ", textnow)

    textnow = re.sub(r"([\w]{1})[\'\`\´\‘\’]([\w]+)", r"\1 _S_\2", textnow)

    return textnow


def tokenizer01(textnow, token_regex=token_pattern_compiled, mapper=None):

    words = token_regex.findall(textnow)

    if mapper is not None:

        words = [mapper[wordnow] if wordnow in mapper else wordnow for wordnow in words]

    return words


def tokenizer02(textnow, token_regex=token_pattern_compiled, mapper=None, vocabulary=None):

    words = token_regex.findall(textnow)

    if mapper is not None:

        words = [mapper[wordnow] if wordnow in mapper else wordnow for wordnow in words]

    if vocabulary is not None:

        words = [wordnow for wordnow in words if wordnow in vocabulary]

    return words


class Vocabulary:

    def __init__(self, docs=None, counts=None, tokenizer=tokenizer01, preprocessor=None):

        self.vocab_reduced = []
        self.word_counts = Counter()
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor

        if docs is not None and counts is not None:
            print("Error", flush=True)
            raise Exception

        if docs is not None:
            for doc in docs:
                self.add_doc(doc)

        if counts is not None:
            self.word_counts.update(counts)

    def add_doc(self, docnow):

        if self.preprocessor is not None:

            docnow = self.preprocessor(docnow)

        words = self.tokenizer(docnow)

        self.word_counts.update(words)

    def reduce_voc(self, n_min_count=None, n_words=None, stop_words=None, rm_startswith=None, mapper=None):

        count = 0
        self.vocab_reduced = []

        for wordnow, wordcountnow in self.word_counts.most_common():

            if n_min_count is not None and wordcountnow < n_min_count:
                break

            if n_words is not None and count == n_words:
                break

            if stop_words is not None and wordnow in stop_words:
                continue

            if rm_startswith is not None and wordnow[:len(rm_startswith)] == rm_startswith:
                continue

            if mapper is not None and wordnow in mapper:
                wordnow = mapper[wordnow]

            self.vocab_reduced.append(wordnow)

            count += 1

        return self.vocab_reduced

    def write_voc_all(self, filepath):
        print("Writing vocabulary to ", filepath, flush=True)
        with open(filepath, 'w') as f:
            for word, count in self.word_counts.most_common():
                f.write(str(count) + '\t' + word + '\n')

    def write_voc_reduced(self, filepath):
        print("Writing reduced vocabulary to ", filepath, flush=True)
        with open(filepath, 'w') as f:
            for word in self.vocab_reduced:
                f.write(str(self.word_counts[word]) + '\t' + word + '\n')

    def get_vocab_reduced_dict(self):

        return {wordnow: indnow for indnow, wordnow in enumerate(self.vocab_reduced)}


def iter_texts(dfnow, textdict=None, idcolname='TextId', preprocessor=None):

    for ind, row in dfnow.iterrows():

        if preprocessor is None:
            yield textdict[row[idcolname]]
        else:
            yield preprocessor(textdict[row[idcolname]])


def count_words(textnow, processor=preprocess_tr, tokenizer=tokenizer01):

    if processor is not None:
        textnow = processor(textnow)

    wordsnow = tokenizer(textnow)

    nwords = len(wordsnow)
    nchars = np.sum([len(wordnow) for wordnow in wordsnow])

    return nwords, nchars


def indexer01(wordsnow, vocabulary_dict):

    inds = [vocabulary_dict[wordnow] for wordnow in wordsnow]

    ind_counts = Counter(inds)

    indexes = [keynow for keynow in ind_counts.keys()]
    counts = [ind_counts[indnow] for indnow in indexes]

    return indexes, counts


def indexer02(wordsnow, vocabulary_dict):

    return [vocabulary_dict[wordnow] for wordnow in wordsnow]
