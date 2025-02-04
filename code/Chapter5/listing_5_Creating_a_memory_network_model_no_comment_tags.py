def create_model(trainingData, testData, context):

    tokenizer,vocab_size=create_tokenizer(trainingData,testData) 

    X_tr,Q_tr,y_tr,max_story_len_tr, max_query_len_tr=
        process_stories_n_context(trainingData,tokenizer,vocab_size,
        use_context=context) 
    X_te,Q_te,y_te, max_story_len_te, max_query_len_te=
        process_stories_n_context(testData,tokenizer,vocab_size,
        use_context=context)

    max_story_len=max(max_story_len_tr,
        max_story_len_te) 
    max_query_len=max(max_query_len_tr, max_query_len_te)

    X_tr, Q_tr=pad_data(X_tr,Q_tr,max_story_len,
        max_query_len) 
    X_te, Q_te=pad_data(X_te,Q_te,max_story_len, max_query_len)

    input = Input((max_story_len,))  
    question = Input((max_query_len,))

    A= Embedding(input_dim=vocab_size,
                              output_dim=64) 
    C=Embedding(input_dim=vocab_size,
                              output_dim=max_query_len)
    B=Embedding(input_dim=vocab_size,
                               output_dim=64,
                               input_length=max_query_len)

    input_A = A(input) 
    input_C = C(input)
    question_B = B(question)

    input_question_match = dot([input_A, question_B],
        axes=(2, 2)) 
    Probs = Activation('softmax')(input_question_match) 

    O = add([Probs, input_C]) 
    O = Permute((2, 1))(O) 

    final_match = concatenate([O, question_B]) 

    size=keras.backend.int_shape(final_match)[2] 
    weights = Dense(size, activation='softmax')
        (final_match) 

    merged=merge([final_match, weights], mode='mul') 
    answer=Flatten()(merged)

    answer = Dense(vocab_size)(answer)  
    answer = Activation('softmax')(answer)

    model = Model([input_sequence, question], answer) 
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])

    return X_tr,Q_tr,y_tr,X_te,Q_te,y_te,model
