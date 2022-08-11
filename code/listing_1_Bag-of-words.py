import pandas as pd  <1>
from sklearn.feature_extraction.text import CountVectorizer <2>

trainingdata = pd.read_csv("train.tsv", header=0, encoding='utf-8',
[CA]delimiter="\t") <3>

cv = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 1000)  <4>

docvec=cv.fit_transform(trainingdata["text"]).toarray() <5>

print docvec <6>

print cv.vocabulary_ <7>

