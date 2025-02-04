def createModel(attention_flag=False, return_sequences=False, timesteps=1,
[CA]input_dim=1, maxlen=64, num_classes=1):
    maxwords= 1000
    maxlen=100
    vlen=maxlen

    model = Sequential()
    model.add(Embedding(maxwords, vlen, input_length=maxlen)) <1>
    model.add(AttentionLSTM(maxlen, <2>
                            return_sequences=return_sequences, <2>
                            attention_flag=attention_flag, <2>
                            input_shape=(maxlen,vlen), <2>
                            dropout=0.2,name='attention_lstm', <2>
                            recurrent_dropout=0.2)) <2>
    model.add(Dense(1, activation='sigmoid')) <3>
    model.compile(loss='binary_crossentropy', <4>
                  optimizer='adam',
                  metrics=['accuracy'])
    return model <5>
