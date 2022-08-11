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


def custom_loss(a,b):
    def loss(y_true,y_pred):
        e1=keras.losses.categorical_crossentropy(y_true,y_pred)
        e2=keras.losses.mean_squared_error(a,b)
        e3=keras.losses.cosine_proximity(a,b)
        e4=K.mean(K.square(a-b), axis=-1)
        return e1+e2+e3+e4
    return loss


model.compile(loss=custom_loss(x1,x2),
              optimizer='adam',
              metrics=['accuracy'])
