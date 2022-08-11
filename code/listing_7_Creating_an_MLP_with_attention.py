inputs = Input(shape=(max_words,)) <1>
attention = Dense(max_words, activation='softmax', name='attention')(
[CA]inputs) <2>

attention_prod = merge([inputs, attention], output_shape=max_words,
[CA]name='attention_prod', mode='mul') <3>

attention_prod = Dense(256)(attention_prod) <4>

attention_prod=Activation('relu')(attention_prod) <5>

output = Dense(num_classes, activation='softmax')(attention_prod) <6>

model = Model(input=[inputs], output=output)
model.compile(loss='categorical_crossentropy', optimizer='adam',
[CA]metrics=['accuracy'])
