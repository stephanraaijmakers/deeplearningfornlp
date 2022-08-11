from keras.models import Sequential  <1>
from keras.layers import Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Convolution1D, Flatten, Dropout
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
import pandas as pd
import sys
from keras.utils.vis_utils import plot_model

data = pd.read_csv(sys.argv[1],sep='\t') <2>
max_words = 1000
tokenizer = Tokenizer(num_words=max_words, split=' ')
tokenizer.fit_on_texts(data['text'].values)

X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)
Y = pd.get_dummies(data['label']).values

X_train, X_test, y_train, y_test = train_test_split(
[CA]X,Y, test_size = 0.2, random_state = 36) <3>

embedding_vector_length = 100

model = Sequential() <4>
model.add(Embedding(
[CA]max_words, embedding_vector_length,
[CA]input_length=X.shape[1])) <5>
model.add(Convolution1D(64, 3, padding="same")) <6>
model.add(Convolution1D(32, 3, padding="same"))
model.add(Convolution1D(16, 3, padding="same"))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(2,activation='sigmoid'))

model.summary() <7>
plot_model(model, to_file='model.png')

model.compile(loss='binary_crossentropy', optimizer='adam',
[CA]metrics=['accuracy']) <8>

model.fit(X_train, y_train, epochs=3, batch_size=64)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
