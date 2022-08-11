(x1_train, y1_train), (x2_train, y2_train), (x1_test,y1_test),(x2_test, y2_test)= load_conll(train, test)

num_classes1=np.max(np.concatenate((y1_train,y1_test),axis=None))+1
num_classes2=np.max(np.concatenate((y2_train,y2_test),axis=None))+1

y1_train = keras.utils.to_categorical(y1_train, num_classes1)
y1_test = keras.utils.to_categorical(y1_test, num_classes1)
y2_train = keras.utils.to_categorical(y2_train, num_classes2)
y2_test = keras.utils.to_categorical(y2_test, num_classes2)
