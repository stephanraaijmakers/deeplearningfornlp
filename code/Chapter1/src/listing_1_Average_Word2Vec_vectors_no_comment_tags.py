import gensim
import numpy as np

tweets=["With the great vote on Cutting Taxes, this could be a big day for
the Stock Market - and YOU","Putting Pelosi/Schumer Liberal Puppet
Jones into office in Alabama would hurt our great Republican Agenda
of low on taxes, tough on crime, strong on military and borders...
& so much more. Look at your 401-k’s since Election. Highest Stock
Market EVER! Jobs are roaring back!",
...] 

model = gensim.models.Word2Vec.load_word2vec_format(
'GoogleNews-vectors-negative300.bin', binary=True)  

vectA=[]

for tweet in tweets:
    vect=np.zeros(300) 
    n=0 
    for word in tweet.split(" "):
        if word in model.wv:
            vect=np.add(vect, model.wv[word]) 
            n+=1
    vect=np.divide(vect,n) 
    vectA.append(vect)

return vectA

