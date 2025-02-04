from keras.models import Sequential
from keras.layers import Embedding
import numpy as np

model = Sequential()
model.add(Embedding(100, 8, input_length=10)) 

input_array = np.random.randint(100, size=(10, 10)) 

model.compile('rmsprop', 'mse') 

output_array = model.predict(input_array) 
