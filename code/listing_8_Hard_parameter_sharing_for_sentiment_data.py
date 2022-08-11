ClassLexicon={} <1>
(x1_train,y1_train),(x1_test,y1_test)=loadData(train1,test1)
num_classes1=len(ClassLexicon)
x1_train = pad_sequences(x1_train, maxlen=max_length, padding='post')
y1_train = keras.utils.to_categorical(y1_train, num_classes1)
x1_test = pad_sequences(x1_test, maxlen=max_length, padding='post')
y1_test = keras.utils.to_categorical(y1_test, num_classes1)

ClassLexicon={} <2>
(x2_train,y2_train),(x2_test,y2_test)=loadData(train2,test2)
num_classes2=len(ClassLexicon)
x2_train = pad_sequences(x2_train, maxlen=max_length, padding='post')
y2_train = keras.utils.to_categorical(y2_train, num_classes2)
x2_test = pad_sequences(x2_test, maxlen=max_length, padding='post')
y2_test = keras.utils.to_categorical(y2_test, num_classes2)

epochs = 100
batch_size=128
max_words=len(Lexicon)+1
max_length = 1000

inputsA=Input(shape=(max_length,)) <3>
x1=Embedding(300000, 16)(inputsA)
x1=Dense(64,activation='relu')(x1)
x1=Dense(32,activation='relu')(x1)
x1=Flatten()(x1)

inputsB=Input(shape=(max_length,)) <4>
x2=Embedding(300000, 16)(inputsB)
x2=Dense(64,activation='relu')(x2)
x2=Flatten()(x2)

merged = Concatenate()([x1, x2]) <5>

y1=Dense(num_classes1,activation='softmax')(merged) <6>
y2=Dense(num_classes2,activation='softmax')(merged)

model=Model(inputs=[inputsA, inputsB],outputs=[y1,y2]) <7>

model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

history = model.fit([x1_train,x2_train], [y1_train,y2_train],
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_split=0.1)

score = model.evaluate([x1_test,x2_test], [y1_test,y2_test],
               batch_size=batch_size, verbose=1)

print(score)
