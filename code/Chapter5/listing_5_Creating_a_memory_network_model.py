def create_model(trainingData, testData, context):

    tokenizer,vocab_size=create_tokenizer(trainingData,testData) <1>

    X_tr,Q_tr,y_tr,max_story_len_tr, max_query_len_tr=
        process_stories_n_context(trainingData,tokenizer,vocab_size,
        use_context=context) <2>
    X_te,Q_te,y_te, max_story_len_te, max_query_len_te=
        process_stories_n_context(testData,tokenizer,vocab_size,
        use_context=context)

    max_story_len=max(max_story_len_tr,
        max_story_len_te) <3>
    max_query_len=max(max_query_len_tr, max_query_len_te)

    X_tr, Q_tr=pad_data(X_tr,Q_tr,max_story_len,
        max_query_len) <4>
    X_te, Q_te=pad_data(X_te,Q_te,max_story_len, max_query_len)

    input = Input((max_story_len,))  <5>
    question = Input((max_query_len,))

    A= Embedding(input_dim=vocab_size,
                              output_dim=64) <6>
    C=Embedding(input_dim=vocab_size,
                              output_dim=max_query_len)
    B=Embedding(input_dim=vocab_size,
                               output_dim=64,
                               input_length=max_query_len)

    input_A = A(input) <7>
    input_C = C(input)
    question_B = B(question)

    input_question_match = dot([input_A, question_B],
        axes=(2, 2)) <8>
    Probs = Activation('softmax')(input_question_match) <9>

    O = add([Probs, input_C]) <10>
    O = Permute((2, 1))(O) <11>

    final_match = concatenate([O, question_B]) <12>

    size=keras.backend.int_shape(final_match)[2] <13>
    weights = Dense(size, activation='softmax')
        (final_match) <14>

    merged=merge([final_match, weights], mode='mul') <15>
    answer=Flatten()(merged)

    answer = Dense(vocab_size)(answer)  <16>
    answer = Activation('softmax')(answer)

    model = Model([input_sequence, question], answer) <17>
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])

    return X_tr,Q_tr,y_tr,X_te,Q_te,y_te,model
