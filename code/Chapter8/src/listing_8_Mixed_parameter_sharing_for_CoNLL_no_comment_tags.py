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

x_a=Flatten()(x_a) 
x_b=Flatten()(x_b)

merged = Concatenate()([x1, x2]) 

y1=Dense(num_classes1, activation='softmax')(merged)
y2=Dense(num_classes2, activation='softmax')(merged)

model=Model(inputs=[inputsA, inputsB],outputs=[y1,y2])
...
model.compile(loss=custom_loss(x_a,x_b), 
              optimizer='adam',
              metrics=['categorical_accuracy'])
...
