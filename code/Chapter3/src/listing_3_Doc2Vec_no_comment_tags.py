from keras.models import Model
from keras.layers import Input, Dense, Reshape, merge, concatenate,
average, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import numpy as np
import sys
import random
import re
from keras.utils import to_categorical
from nltk.util import ngrams
import codecs


def save_embedding(outputFile, weights, nb_docs):
    with codecs.open(outputFile, "w", "utf-8") as f:
        f.write(str(nb_docs) + " " + str(weights.shape[1]) + "\n")
                for index in range(nb_docs):
            f.write("doc_"+str(index) + " ")
                        for i in xrange(len(weights[index])):
                    f.write(str(weights[index][i]) + " ")
                    f.write("\n")

def getLines(f):
    lines = [line.rstrip() for line in open(f)]
    return lines


def create_vocabulary(vocabulary, docs):
        vocabulary["<unk>"]=0
    for doc in docs:
        for word in doc.strip().split():
                        word=re.sub("[.,:;'\"!?()]+","",word.lower())
                        if word not in vocabulary:
                    vocabulary[word]=len(vocabulary)


def generator(contexts, targets, batch_size):
    w1 = np.zeros((batch_size, 1))
    w2 = np.zeros((batch_size, 1))
    w3 = np.zeros((batch_size, 1))
    docid = np.zeros((batch_size, 1))
    batch_targets = np.zeros((batch_size,1))

    while True:
        for i in range(batch_size):
            index= random.randint(0,len(targets)-1)
            batch_targets[i] = targets[index]
            docid[i]=contexts[index][0]
            w1[i] = contexts[index][1]
            w2[i] = contexts[index][2]
            w3[i] = contexts[index][3]
        yield [w1,w2,w3,docid], [batch_targets]



def process_data(textFile,window_size):
    docs = getLines(textFile)
    vocab = dict()
    create_vocabulary(vocab, docs)
    docid=0
    contexts=[]
    docids=[]
    targets=[]

    f=open("docs.legenda","w")
    for s in docs:
        f.write("%d %s\n"%(docid,s))
        docids.append(docid)
        ngs=list(ngrams(s.split(), window_size))
        for i in range(len(ngs)-1):
            cs=[docid]
            ng=ngs[i]
            for w in ng:
                w=re.sub("[.,:;'\"!?()]+","",w.lower())
                cs.append(vocab[w])
            contexts.append(cs)
            target_word=re.sub("[.,:;'\"!?()]+","",ngs[i+1][0].lower())
            targets.append(vocab[target_word])
        docid+=1
    f.close()
    return np.array(contexts),np.array(docids),np.array(targets),vocab



window_size = 3
vector_dim = 100
epochs = 1000

contexts,docids,targets,vocab=collect_data(sys.argv[1],3)
vocab_size=len(vocab)

input_w1 = Input((1,))
input_w2 = Input((1,))
input_w3 = Input((1,))
input_docid=Input((1,))

embedding = Embedding(vocab_size, vector_dim, input_length=1,
name='embedding')

vector_dim_doc=vector_dim
embedding_doc=Embedding(len(docids)+1,vector_dim_doc)

docid=embedding_doc(input_docid)
docid = Reshape((vector_dim_doc,1))(docid)

w1 = embedding(input_w1)
w1 = Reshape((vector_dim, 1))(w1)

w2 = embedding(input_w2)
w2 = Reshape((vector_dim, 1))(w2)

w3 = embedding(input_w3)
w3 = Reshape((vector_dim, 1))(w3)

context_docid=concatenate([w1,w2,w3,docid])
context_docid=Flatten()(context_docid)
output = Dense(vocab_size,activation='softmax')(context_docid)
model = Model(input=[input_w1, input_w2, input_w3, input_docid],
output=output)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['acc'])

print(model.summary())

epochs=int(sys.argv[2])

model.fit_generator(generator(contexts,targets,100), steps_per_epoch=100,
epochs=epochs)

save_embeddings("embedding_doc2vec.txt", embedding_doc.get_weights()[0],
len(docids))

exit(0)
