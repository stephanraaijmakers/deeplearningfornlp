from keras.models import Model
from keras.layers import Input, Dense, Reshape, merge
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence

import numpy as np
import sys

import random
import re
import codecs

def save_embedding(outputFile, weights, vocabulary):
  rev = {v:k for k, v in vocabulary.iteritems()}
 with codecs.open(outputFile, "w") as f:
  f.write(str(len(vocabulary)) + " " + str(weights.shape[1]) + "\n")
    for index in sorted(rev.iterkeys()):
            word=rev[index]
         f.write(word + " ")
            for i in range(len(weights[index])):
             f.write(str(weights[index][i]) + " ")
           f.write("\n")

def getLines(f):
    lines = [line.rstrip() for line in open(f)]
    return lines

def generator(target,context, labels, batch_size):
    batch_target = np.zeros((batch_size, 1))
    batch_context = np.zeros((batch_size, 1))
    batch_labels = np.zeros((batch_size,1))

    while True:
        for i in range(batch_size):
            index= random.randint(0,len(target)-1)
            batch_target[i] = target[index]
            batch_context[i]=context[index]
            batch_labels[i] = labels[index]
        yield [batch_target,batch_context], [batch_labels]

def process_data(textFile,window_size): 
    couples=[]
    labels=[]
    sentences = getLines(textFile)
    vocab = dict()
    create_vocabulary(vocab, sentences)
    vocab_size=len(vocab)
    for s in sentences:
        words=[]
        for w in s.split(" "):
            w=re.sub("[.,:;'\"!?()]+","",w.lower())
            if w!='':
                words.append(vocab[w])
        c,l=skipgrams(words,vocab_size,window_size=window_size)
        couples.extend(c)
        labels.extend(l)
    return vocab,couples,labels

def create_vocabulary(vocabulary, sentences):
  vocabulary["<unk>"]=0
 for sentence in sentences:
  for word in sentence.strip().split():
        word=re.sub("[.,:;'\"!?()]+","",word.lower())
        if word not in vocabulary:
           vocabulary[word]=len(vocabulary)

window_size = 3
vector_dim = 100
epochs = 1000

vocab,couples,labels=process_data(sys.argv[1],window_size)

vocab_size=len(vocab)

word_target, word_context = zip(*couples) 

input_target = Input((1,)) 
input_context = Input((1,))

embedding = Embedding(vocab_size, vector_dim,
input_length=1) 
target = embedding(input_target)
target = Reshape((vector_dim, 1))(target)
context = embedding(input_context)
context = Reshape((vector_dim, 1))(context)

dot_product = merge([target, context], mode='dot',
dot_axes=1) 
dot_product = Reshape((1,))(dot_product)
output = Dense(1, activation='sigmoid')(dot_product)
model = Model(input=[input_target, input_context], output=output)
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc'])

print(model.summary())

epochs=int(sys.argv[2])

model.fit_generator(generator(word_target, word_context,labels,100),
steps_per_epoch=100, epochs=epochs) 

save_embedding("embedding.txt",
embedding.get_weights()[0], vocab) 
