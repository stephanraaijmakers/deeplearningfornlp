inputsA=Input(shape=(max_length,))
x_a=Embedding(num_words, embedding_vector_length)(inputsA)
x1=Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x_a)
x1=MaxPooling1D(pool_size=2)(x1)
x1=LSTM(100)(x1)

inputsB=Input(shape=(max_length,)) 
x_b=Embedding(num_words, embedding_vector_length)(inputsB)
x2=Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x_b)
x2=MaxPooling1D(pool_size=2)(x2)
x2=LSTM(100)(x2)

y1=Dense(num_classes1, activation='softmax')(x1)
y2=Dense(num_classes2, activation='softmax')(x2)

model=Model(inputs=[inputsA, inputsB],outputs=[y1,y2])

x_a=Flatten()(x_a) 
x_b=Flatten()(x_b)

def custom_loss(a,b): 
    def loss(y_true,y_pred):
      e1=keras.losses.categorical_crossentropy(y_true,y_pred)
      e2=keras.losses.mean_squared_error(a,b)
      e3=keras.losses.cosine_proximity(a,b)
      e4=K.mean(K.square(a-b), axis=-1)
      return e1+e2+e3+e4
    return loss

model.compile(
          loss=custom_loss(x_a,x_b), 
          optimizer='adam',
          metrics=['categorical_accuracy'])

history = model.fit([x1_train,x2_train], [y1_train,y2_train], 
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_split=0.1)

score = model.evaluate([x1_test,x2_test], [y1_test,y2_test], 
                       batch_size=batch_size, verbose=1)
