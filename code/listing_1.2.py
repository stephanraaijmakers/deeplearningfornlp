from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.layers.core import Dense, Activation

import pandas as pd 
import sys

data = pd.read_csv(sys.argv[1],sep='\t') 
docs=data["text"] 

tokenizer = Tokenizer() 
tokenizer.fit_on_texts(docs) 

X_train = tokenizer.texts_to_matrix(docs, mode='binary') 
y_train=np_utils.to_categorical(data["label"])

input_dim = X_train.shape[1] 
nb_classes = y_train.shape[1]

model = Sequential()
model.add(Dense(128, input_dim=input_dim))
model.add(Activation('sigmoid')) 
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dense(nb_classes)) 
model.add(Activation('softmax')) 
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy']) 

print("Training...")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, shuffle=False,verbose=2) 

