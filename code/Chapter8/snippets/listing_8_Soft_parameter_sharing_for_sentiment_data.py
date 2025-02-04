inputsA=Input(shape=(max_length,))
x1=Embedding(300000, 16)(inputsA)
x1=Dense(64,activation='relu')(x1)
x1=Flatten()(x1)

inputsB=Input(shape=(max_length,))
x2=Embedding(300000, 16)(inputsB)
x2=Dense(64,activation='relu')(x2)
x2=Flatten()(x2)

y1=Dense(num_classes1,activation='softmax')(x1)
y2=Dense(num_classes2,activation='softmax')(x2)

model=Model(inputs=[inputsA, inputsB],outputs=[y1,y2])

model.compile(loss=custom_loss(x1,x2),
              optimizer='adam',
              metrics=['accuracy'])
