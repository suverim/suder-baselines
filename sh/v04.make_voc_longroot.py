#!/usr/bin/python3

from collections import Counter
import sys
import pickle

sys.path.append('../bin')
import utils04 as utils

invocfile_sabah = '../data/sabah/vocab01.txt'
mapperfile_sabah = '../data/sabah/vocab01_root_mapper01.txt'
outvocfile_sabah = '../data/sabah/vocab02_longroot.txt'
outvocfile_sabah_pickle = '../data/sabah/vocab02_longroot.pickle'

invocfile_cumhuriyet = '../data/cumhuriyet/vocab01.txt'
mapperfile_cumhuriyet = '../data/cumhuriyet/vocab01_root_mapper01.txt'
outvocfile_cumhuriyet = '../data/cumhuriyet/vocab02_longroot.txt'
outvocfile_cumhuriyet_pickle  = '../data/cumhuriyet/vocab02_longroot.pickle'

word_mapper = {}
with open(mapperfile_sabah) as f:
    for line in f:
        key, val = line.strip().split(' ')
        word_mapper[key] = val

outvoc = Counter()
with open(invocfile_sabah) as f:
    for line in f:
        countnow, wordnow = line.strip().split('\t')
        countnow = int(countnow)

        if wordnow not in word_mapper:
            print(wordnow,"not in mapper")
            outvoc.update({wordnow: countnow})
        else:
            outvoc.update({word_mapper[wordnow]: countnow})

with open(outvocfile_sabah, 'w') as f:
    for wordnow, countnow in outvoc.most_common():
        f.write(str(countnow) + '\t' + wordnow + '\n')

vocabulary = utils.Vocabulary(counts=outvoc)

with open(outvocfile_sabah_pickle, 'wb') as f:
    pickle.dump(vocabulary, f)



word_mapper = {}
with open(mapperfile_cumhuriyet) as f:
    for line in f:
        key, val = line.strip().split(' ')
        word_mapper[key] = val

outvoc = Counter()
with open(invocfile_cumhuriyet) as f:
    for line in f:
        countnow, wordnow = line.strip().split('\t')
        countnow = int(countnow)

        if wordnow not in word_mapper:
            print(wordnow,"not in mapper")
            outvoc.update({wordnow: countnow})
        else:
            outvoc.update({word_mapper[wordnow]: countnow})

with open(outvocfile_cumhuriyet, 'w') as f:
    for wordnow, countnow in outvoc.most_common():
        f.write(str(countnow) + '\t' + wordnow + '\n')

vocabulary = utils.Vocabulary(counts=outvoc)

with open(outvocfile_cumhuriyet_pickle, 'wb') as f:
    pickle.dump(vocabulary, f)
