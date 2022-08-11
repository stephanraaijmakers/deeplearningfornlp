inputs=Input(shape=(max_length,)) 
x=Embedding(num_words, embedding_vector_length)(inputs) 
x=Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x) 
x=MaxPooling1D(pool_size=2)(x) 
x=LSTM(100)(x) 
y=Dense(num_classes1, activation='softmax')(x) 

model=Model(inputs=inputsA,outputs=y) 
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy'])
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)

score = model.evaluate(x_test, y_test, 
                      batch_size=batch_size, verbose=1)
