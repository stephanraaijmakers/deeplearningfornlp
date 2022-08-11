(x_train,y_train),(x_test,y_test)=loadData(train,test) <1>
num_classes=len(ClassLexicon) 

epochs = 100
batch_size=128

max_words=len(Lexicon)+1

max_length = 1000
x_train = pad_sequences(x_train, maxlen=max_length,
          padding='post') <2>
x_test = pad_sequences(x_test, maxlen=max_length, padding='post')

y_train = keras.utils.to_categorical(y_train, num_classes) <3>
y_test = keras.utils.to_categorical(y_test, num_classes)

inputs=Input(shape=(max_length,)) <4>
x=Embedding(300000, 16)(inputs) <5>
x=Dense(64,activation='relu')(x) <6>
x=Flatten()(x) <7>
y=Dense(num_classes,activation='softmax')(x) <8>

model=Model(inputs=inputs, outputs=y) <9>
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)
