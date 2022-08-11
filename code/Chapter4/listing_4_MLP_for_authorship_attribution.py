model = Sequential()
model.add(Embedding(vocab_len, 300, input_length=input_dim))
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(nb_classes, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam',
[CA]metrics=['acc'])

print model.summary()

nb_epochs=10

model.fit(X_train, y_train, epochs=nb_epochs, shuffle=True,batch_size=64,
[CA]validation_split=0.3, verbose=2)
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print('Accuracy: %f' % (accuracy*100))
