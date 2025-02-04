inputs=Input(shape=(max_length,)) <1>

x=Embedding(num_words, embedding_vector_length)(inputs) <2>

x=Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x) <3>

x=MaxPooling1D(pool_size=2)(x) <4>

x=LSTM(100)(x) <5>

y=Dense(num_classes1, activation='softmax')(x) <6>

model=Model(inputs=inputsA,outputs=y) <7>
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy'])
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)

score = model.evaluate(x_test, y_test, <7>
                      batch_size=batch_size, verbose=1)
