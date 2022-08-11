
left_input = Input(shape=(input_dim,), dtype='int32')
right_input = Input(shape=(input_dim,), dtype='int32')

embedding = Embedding(vocab_size, 300, input_length=input_dim)
encoded_left = embedding(left_input) 
encoded_right = embedding(right_input)

nb_units=10 

lstm = LSTM(nb_units) 
left_output = lstm(encoded_left)
right_output = lstm(encoded_right)

model_distance = Lambda(function=lambda x: exp_neg_manhattan_distance(x[0], x[1]),output_shape=lambda x: (x[0][0], 1))([left_output, right_output]) 

model = Model([left_input, right_input], [model_distance]) 

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit([X_train_left, X_train_right], y_train_lr, batch_size=64, nb_epoch=nb_epochs,
                            validation_split=0.3, verbose=2)
model.evaluate([X_test_left, X_test_right], y_test_lr)

loss, accuracy = model.evaluate([X_test_left, X_test_right], y_test_lr, verbose=0)
print('Accuracy: %f' % (accuracy*100))
