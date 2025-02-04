def custom_loss(a,b):
        def loss(y_true,y_pred):
            e1=keras.losses.categorical_crossentropy(y_true,y_pred)
            e2=keras.losses.mean_squared_error(a,b)
            e3=keras.losses.cosine_proximity(a,b)
            e4=K.mean(K.square(a-b), axis=-1)
            return e1+e2+e3+e4
        return loss
