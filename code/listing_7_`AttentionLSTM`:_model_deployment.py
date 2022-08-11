x_train,y_train,x_test,y_test, RevWordIndex,num_classes, TEST = createData()

maxlen=max([max([len(x) for x in x_train]),max([len(x) for x in x_test])])
input_dim=100 # words
timesteps=10
batch_size = 32

model1=createModel(attention_flag=False,return_sequences=False, <1>
[CA]timesteps=timesteps,input_dim=input_dim,maxlen=maxlen, <1>
[CA]num_classes=num_classes) <1>

model1.fit(x_train,y_train, <2>
          batch_size=batch_size,
          epochs=5
          )
model1.save_weights("m1.weights.h5") <3>
score = model1.evaluate(x_test, y_test, batch_size=batch_size, verbose=1) <4>
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model2=createModel(attention_flag=True, return_sequences=True, <5>
[CA]timesteps=timesteps, input_dim=input_dim,maxlen=maxlen, <5>
[CA]num_classes=num_classes) <5>

model2.load_weights("m1.weights.h5") <6>

TEST_DATA=... # a test document from x_test
attention=getLayerActivation( <7>
[CA]model2, TEST_DATA,layerName='attention_lstm')[0] <7>
discrete_attention=[]
for window in attention:
      normalized_window=window/sum(window)
      discrete_window=discretize_attention(normalized_window)
      discrete_attention.append(discrete_window)
...
