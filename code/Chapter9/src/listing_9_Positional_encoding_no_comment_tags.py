import sys
from math import sin, cos
import numpy as np

dimension=4
max_len=10
pos_enc=np.zeros(shape=(max_len,dimension)) 
for pos in range(max_len):
   i=0
   while (i<=dimension-2): 
      x = pos/(10000**(2.0*i/dimension))
      pos_enc[pos][i] = sin(x)
      pos_enc[pos][i+1] = cos(x)
      i+=2
print(pos_enc) 
