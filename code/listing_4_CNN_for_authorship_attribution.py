model = Sequential()
model.add(Embedding(vocab_size, 300, input_length=input_dim))
model.add(Dense(300, activation='relu'))
model.add(Convolution1D(32, 30, padding="same"))
model.add(Flatten())
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',
[CA]metrics=['acc'])

print model.summary()

model.fit(X_train, y_train, epochs=nb_epochs, shuffle=True,batch_size=16,
[CA]validation_split=0.3, verbose=2)
