from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plotTSNE(bert_vectors, text): 
    tsne = TSNE(n_components=2, init='pca') 
    output = tsne.fit_transform(bert_vectors)

    x_vals = [] 
    y_vals = []
    for xy in output:
        x_vals.append(xy[0])
        y_vals.append(xy[1])

    plt.figure(figsize=(5, 5)) 

    words=text.split(" ") 
    for i in range(len(x_vals)):
        plt.scatter(x_vals[i],y_vals[i])
        plt.annotate(words[i],
                     xy=(x_vals[i], y_vals[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig('bert.png') 

