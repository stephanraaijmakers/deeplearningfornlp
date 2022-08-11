
from wordcloud import WordCloud 
import matplotlib.pyplot as plt


# Process attention
P={}
n=0

for attval in discrete_attention: 
    word_id=DOCUMENTS[0][n] 
    if RevWordIndex[word_id] in P: <4>
            P[RevWordIndex[word_id]]+=attval
        else:
            P[RevWordIndex[word_id]]=attval
    n+=1

# Use ordinal positions
n=1
Q={}
for w in sorted(P, key=P.get, reverse=False): 
    Q[w]=n
    n+=1

wc = WordCloud(background_color="white", max_words=1000).generate_from_frequencies(Q) 

plt.imshow(wc, interpolation='bilinear') 
plt.axis("off")
plt.show()
