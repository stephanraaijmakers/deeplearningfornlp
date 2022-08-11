from sklearn.linear_model import Perceptron <1>
from sklearn.datasets import fetch_20newsgroups <2>

categories = ['alt.atheism', 'sci.med'] <3>

train = fetch_20newsgroups(
[CA]subset='train',categories=categories, shuffle=True) <4>

perceptron = Perceptron(max_iter=100) <5>

from sklearn.feature_extraction.text import CountVectorizer <6>
cv = CountVectorizer()
X_train_counts = cv.fit_transform(train.data)

from sklearn.feature_extraction.text import TfidfTransformer <7>
tfidf_tf = TfidfTransformer()
X_train_tfidf = tfidf_tf.fit_transform(X_train_counts)

perceptron.fit(X_train_tfidf,train.target) <8>

test_docs = ['Religion is widespread, even in modern times', 'His kidney
[CA]failed','The pope is a controversial leader', 'White blood cells fight
[CA]off infections','The reverend had a heart attack in church'] <9>

X_test_counts = cv.transform(test_docs) <10>
X_test_tfidf = tfidf_tf.transform(X_test_counts)

pred = perceptron.predict(X_test_tfidf) <11>

for doc, category in zip(test_docs, pred): <12>
    print('%r => %s' % (doc, train.target_names[category]))
