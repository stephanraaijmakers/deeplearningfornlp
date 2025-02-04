from keras.models import Sequential
from keras.layers import Embedding,Dense,Flatten
import numpy as np
import random
import re
import sys
import codecs
from keras.preprocessing.sequence import pad_sequences

def save_embedding(outputFile, weights, vocabulary): 
  rev = {v:k for k, v in vocabulary.iteritems()}
 with codecs.open(outputFile, "w") as f:
  f.write(str(len(vocabulary)) + " " + str(weights.shape[1]) + "\n")
                for index in sorted(rev.iterkeys()):
                        word=rev[index]
   f.write(word + " ")
                       for i in xrange(len(weights[index])):
              f.write(str(weights[index][i]) + " ")
                 f.write("\n")

def getLines(f): 
    lines = [line.rstrip() for line in open(f)]
    return lines

def create_vocabulary(vocabulary, sentences): 
  vocabulary["<unk>"]=0
 for sentence in sentences:
  for word in sentence.strip().split():
        word=re.sub("[.,:;'\"!?()]+","",word.lower())
        if word not in vocabulary:
       vocabulary[word]=len(vocabulary)

def process_training_data(textFile,max_len): 
    data=[]
    sentences = getLines(textFile)
    vocab = dict()
    labels=[]
    create_vocabulary(vocab, sentences)
    for s in sentences:
        words=[]
        m=re.match("^([^\t]+)\t(.+)$",s.rstrip())
        if m:
            sentence=m.group(1)
            labels.append(int(m.group(2)))
        for w in sentence.split(" "):
            w=re.sub("[.,:;'\"!?()]+","",w.lower())
            if w!='':
                words.append(vocab[w])
        data.append(words)
    data = pad_sequences(data, maxlen=max_len, padding='post')

    return data,labels, vocab

def process_test_data(textFile,vocab,max_len): 
    data=[]
    sentences = getLines(textFile)
    labels=[]
    create_vocabulary(vocab, sentences)
    for s in sentences:
        words=[]
        m=re.match("^([^\t]+)\t(.+)$",s.rstrip())
        if m:
            sentence=m.group(1)
            labels.append(int(m.group(2)))
        for w in sentence.split(" "):
            w=re.sub("[.,:;'\"!?()]+","",w.lower())
            if w!='':
                if w in vocab:
                    words.append(vocab[w])
                else:
                    words.append(vocab["<unk>"])
        data.append(words)
    data = pad_sequences(data, maxlen=max_len, padding='post')
    return data,labels

max_len=100
data,labels,vocab=process_training_data(sys.argv[1],max_len) 
test_data,test_labels=process_test_data(sys.argv[2],vocab,max_len)

model = Sequential()
embedding=Embedding(len(vocab), 100, input_length=max_len) 
model.add(embedding)
model.add(Flatten())
model.add(Dense(1,activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc'])
model.fit(data,labels,epochs=100, verbose=1)

loss, accuracy = model.evaluate(test_data, test_labels, verbose=0)
print accuracy

save_embedding("embedding_labeled.txt",embedding.get_weights()[0], vocab) 

