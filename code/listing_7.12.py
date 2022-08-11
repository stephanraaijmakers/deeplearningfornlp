
def createModel(attention_flag=False, return_sequences=False, timesteps=1, input_dim=1, maxlen=64, num_classes=1):
    maxwords= 1000
    maxlen=100
    vlen=maxlen

    model = Sequential()
    model.add(Embedding(maxwords, vlen, input_length=maxlen)) 
    model.add(AttentionLSTM(maxlen,
                            return_sequences=return_sequences,
                            attention_flag=attention_flag,
                            input_shape=(maxlen,vlen),
                            dropout=0.2,name='attention_lstm',
                            recurrent_dropout=0.2)) 
    model.add(Dense(1, activation='sigmoid')) 
    model.compile(loss='binary_crossentropy', 
                  optimizer='adam',
                  metrics=['accuracy'])
    return model 
