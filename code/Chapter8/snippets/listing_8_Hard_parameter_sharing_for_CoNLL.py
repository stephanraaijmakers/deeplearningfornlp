inputsA=Input(shape=(max_length,)) <1>
x2=Embedding(num_words, embedding_vector_length)(inputsA)
x1=Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x2)
x1=MaxPooling1D(pool_size=2)(x1)
x1=LSTM(100)(x1)

inputsB=Input(shape=(max_length,)) <2>
x2=Embedding(num_words, embedding_vector_length)(inputsB)
x2=Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x2)
x2=MaxPooling1D(pool_size=2)(x2)
x2=LSTM(100)(x2)

merged = Concatenate()([x1, x2]) <3>

y1=Dense(num_classes1, activation='softmax')(merged) <4>
y2=Dense(num_classes2, activation='softmax')(merged)

model=Model(inputs=[inputsA, inputsB],outputs=[y1,y2]) <5>

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
                  metrics=['categorical_accuracy'])

history = model.fit([x1_train,x2_train], [y1_train,y2_train], <6>
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_split=0.1)

score = model.evaluate([x1_test,x2_test], [y1_test,y2_test], <7>
               batch_size=batch_size, verbose=1)
