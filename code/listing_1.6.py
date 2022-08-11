
def feat_hash(featureV ,vecSize):
     outputV=numpy.array(vecSize)
     for f in range(0,len(featureV)):
         if featureV[f]==1:
             dim=hash_function(InverseLexicon[f])
             outputV[dim mod vecSize] += 1
     return outputV
