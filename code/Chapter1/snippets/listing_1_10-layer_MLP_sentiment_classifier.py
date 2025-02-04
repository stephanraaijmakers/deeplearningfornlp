from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.layers.core import Dense, Activation

import pandas as pd <1>
import sys

data = pd.read_csv(sys.argv[1],sep='\t') <2>
docs=data["text"] <3>

tokenizer = Tokenizer() <4>
tokenizer.fit_on_texts(docs) <5>

X_train = tokenizer.texts_to_matrix(docs, mode='binary') <6>
y_train=np_utils.to_categorical(data["label"])

input_dim = X_train.shape[1] <7>
nb_classes = y_train.shape[1]

model = Sequential()
model.add(Dense(128, input_dim=input_dim))
model.add(Activation('sigmoid')) <8>
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
model.add(Dense(nb_classes)) <9>
model.add(Activation('softmax')) <10>
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy']) <11>

print("Training...")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1,
[CA]shuffle=False,verbose=2) <12>

