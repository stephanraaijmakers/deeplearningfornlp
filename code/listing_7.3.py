inputs = Input(shape=(max_words,)) 
attention = Dense(max_words, activation='softmax', name='attention')(inputs) 
attention_prod = merge([inputs, attention], output_shape=max_words, name='attention_prod', mode='mul') 
attention_prod = Dense(256)(attention_prod) 
attenton_prod=Activation('relu')(attention_prod) 
output = Dense(num_classes, activation='softmax')(attention_prod) 

model = Model(input=[inputs], output=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
