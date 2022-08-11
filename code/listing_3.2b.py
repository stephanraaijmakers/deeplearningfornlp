from keras.models import Sequential
from keras.layers import Embedding
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def tsne_plot(model,max_words=100):
    labels = []
    tokens = []

    n=0
    for word in model:
        if n<max_words:
            tokens.append(model[word])
            labels.append(word)
            n+=1


    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=10000, random_state=23) 
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:  
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(8, 8))
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

# ... The document preprocessing from Listing 3.2a.

model = Sequential() 
model.add(Embedding(100, 8, input_length=10))
input_array = np.random.randint(100, size=(10, 10))
model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)

M={}
for i in range(len(input_array)):
    for j in range(len(input_array[i])):
        M[input_array[i][j]]=output_array[i][j]

tsne_plot(M)

