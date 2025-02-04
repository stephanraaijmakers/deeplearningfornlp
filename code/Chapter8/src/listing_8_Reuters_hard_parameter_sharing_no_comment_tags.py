inputsA=Input(shape=(max_length,)) 
x1=Embedding(300000, 16)(inputsA)
x1=Dense(64,activation='relu')(x1)
x1=Flatten()(x1)

inputsB=Input(shape=(max_length,)) 
x2=Embedding(300000, 16)(inputsB)
x2=Dense(64,activation='relu')(x2)
x2=Flatten()(x2)

merged = Concatenate()([x1, x2]) 

y1=Dense(num_classes1,activation='softmax')(merged)
y2=Dense(num_classes2,activation='softmax')(merged)

model=Model(inputs=[inputsA, inputsB],outputs=[y1,y2])

history = model.fit([x1_train,x2_train], [y1_train,y2_train], 
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=0,
                            validation_split=0.1)

score = model.evaluate([x1_test,x2_test], [y1_test,y2_test], 
                               batch_size=batch_size, verbose=1)
