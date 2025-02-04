
def IB1(a,b):
  return sum( [delta(a[i],b[i]) for i in range(len(a))])

def delta(x,y):
   if x==y:
     return 0
   if x!=y:
     return 1

