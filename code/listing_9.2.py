import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

x=range(dimension)
y=range(max_len)
x, y = np.meshgrid(x,y) 

plt.xticks(range(dimension))
plt.pcolormesh(x, y, pos_enc ,cmap=cm.gray) 
plt.colorbar()
plt.xlabel('Embedding dimension')
plt.ylabel('Word position in sequence')
plt.show()
