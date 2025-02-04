from keras.models import Sequential
from keras.layers import Embedding
import numpy as np

model = Sequential()
model.add(Embedding(100, 8, input_length=10)) <1>

input_array = np.random.randint(100, size=(10, 10)) <2>

model.compile('rmsprop', 'mse') <3>

output_array = model.predict(input_array) <4>
