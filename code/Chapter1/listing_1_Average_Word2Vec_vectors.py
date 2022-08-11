import gensim
import numpy as np

tweets=["With the great vote on Cutting Taxes, this could be a big day for
[CA]the Stock Market - and YOU","Putting Pelosi/Schumer Liberal Puppet
[CA]Jones into office in Alabama would hurt our great Republican Agenda
[CA]of low on taxes, tough on crime, strong on military and borders...
[CA]& so much more. Look at your 401-k’s since Election. Highest Stock
[CA]Market EVER! Jobs are roaring back!",
...] <1>

model = gensim.models.Word2Vec.load_word2vec_format(
[CA]'GoogleNews-vectors-negative300.bin', binary=True)  <2>

vectA=[]

for tweet in tweets:
    vect=np.zeros(300) <3>
    n=0 <4>
    for word in tweet.split(" "):
        if word in model.wv:
            vect=np.add(vect, model.wv[word]) <5>
            n+=1
    vect=np.divide(vect,n) <6>
    vectA.append(vect)

return vectA

