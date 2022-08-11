
from wordcloud import WordCloud <1>
import matplotlib.pyplot as plt

# Process attention
P={}
n=0

for attval in discrete_attention: <2>
    word_id=DOCUMENTS[0][n] <3>
    if RevWordIndex[word_id] in P: <4>
            P[RevWordIndex[word_id]]+=attval
        else:
            P[RevWordIndex[word_id]]=attval
    n+=1

# Use ordinal positions
n=1
Q={}
for w in sorted(P, key=P.get, reverse=False): <5>
    Q[w]=n
    n+=1

wc = WordCloud(background_color="white",
[CA]max_words=1000).generate_from_frequencies(Q) <6>

plt.imshow(wc, interpolation='bilinear') <7>
plt.axis("off")
plt.show()
