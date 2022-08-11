import pandas as pd  
from sklearn.feature_extraction.text import CountVectorizer 

trainingdata = pd.read_csv("train.tsv", header=0, encoding='utf-8', delimiter="\t") 

cv = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 1000)  

docvec=cv.fit_transform(trainingdata["text"]).toarray() 

print docvec 

print cv.vocabulary_ 

