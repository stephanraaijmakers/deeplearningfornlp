from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense <1>
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np

np.random.seed(1234)

data = ['xyzaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaxyz',
       'pqraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaapqr']

test_data = ['xyzaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaxyz',
            'pqraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaapqr'] <2>

enc = LabelEncoder()
alphabet = np.array(list(set([c for w in data for c in w])))
enc.fit(alphabet)
int_enc=enc.fit_transform(alphabet)
onehot_encoder = OneHotEncoder(sparse=False)
int_enc=int_enc.reshape(len(int_enc), 1)
onehot_encoded = onehot_encoder.fit_transform(int_enc)

X_train=[]
y_train=[]

for w in data:
    for i in range(len(w)-1):
        X_train.extend(onehot_encoder.transform([enc.transform([w[i]])]))
        y_train.extend(onehot_encoder.transform([enc.transform([w[i+1]])]))

X_test=[]
y_test=[]

for w in test_data:
    for i in range(len(w)-1):
        X_test.extend(onehot_encoder.transform([enc.transform([w[i]])]))
        print i,w[i],onehot_encoder.transform([enc.transform([w[i]])])
        y_test.extend(onehot_encoder.transform([enc.transform([w[i+1]])]))

sample_size=512
sample_len=len(X_train)

X_train = np.array([X_train*sample_size]).reshape(
[CA]sample_size,sample_len,len(alphabet))
y_train = np.array([y_train*sample_size]).reshape(
[CA]sample_size,sample_len,len(alphabet))

test_len=len(X_test)
X_test= np.array([X_test]).reshape(1,test_len,len(alphabet))
y_test= np.array([y_test]).reshape(1,test_len,len(alphabet))

model=Sequential() <3>
model.add(LSTM(input_dim  = len(alphabet), output_dim = 100,
[CA]return_sequences = True))
model.add(TimeDistributed(Dense(output_dim = len(alphabet),
[CA]activation  =  "sigmoid")))
model.compile(loss="binary_crossentropy",metrics=["accuracy"],
[CA]optimizer = "adam")

n=1
while True: <4>
        score = model.evaluate(X_test, y_test, batch_size=32)
        print "[Iteration %d] score=%f"%(n,score[1])
        if score[1] == 1.0:
            break
        n+=1
        model.fit(X_train, y_train, nb_epoch = 1, batch_size = 32)

preds=model.predict(X_test)[0]
for p in preds:
    m=np.argmax(p)
    print(enc.inverse_transform(m))

print(model.evaluate(X_test,y_test,batch_size=32))
