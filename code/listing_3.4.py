import sys
import numpy as np
from gensim.models import KeyedVectors 
import gensim.models
import os

w2v = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(sys.argv[1]), binary=False,unicode_errors='ignore') 

for w in  sorted(w2v.wv.vocab):
    print w,w2v.most_similar(w,topn=3) 
