from sklearn.linear_model import perceptron 
from sklearn.datasets import fetch_20newsgroups 

categories = ['alt.atheism', 'sci.med'] 

train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True) 

perceptron = perceptron.Perceptron(max_iter=100) 

from sklearn.feature_extraction.text import CountVectorizer 
cv = CountVectorizer()
X_train_counts = cv.fit_transform(train.data)

from sklearn.feature_extraction.text import TfidfTransformer 
tfidf_tf = TfidfTransformer()
X_train_tfidf = tfidf_tf.fit_transform(X_train_counts)

perceptron.fit(X_train_tfidf,train.target) 

test_docs = ['Religion is widespread, even in modern times', 'His kidney failed','The pope is a controversial leader', 'White blood cells fight off infections','The reverend had a heart attack in church'] 

X_test_counts = cv.transform(test_docs) 
X_test_tfidf = tfidf_tf.transform(X_test_counts)

pred = perceptron.predict(X_test_tfidf) 

for doc, category in zip(test_docs, pred): 
    print('%r => %s' % (doc, train.target_names[category]))
