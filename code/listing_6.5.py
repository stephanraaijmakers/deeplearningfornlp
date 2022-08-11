def create_model(trainingData, testData, context):
    tokenizer,vocab_size=create_tokenizer(trainingData,testData)

    X_tr,Q_tr,y_tr,max_story_len_tr, max_query_len_tr=process_stories_n_context(trainingData,tokenizer,vocab_size,use_context=context)
    X_te,Q_te,y_te, max_story_len_te, max_query_len_te=process_stories_n_context(testData,tokenizer,vocab_size,use_context=context)

    max_story_len=max(max_story_len_tr, max_story_len_te)
    max_query_len=max(max_query_len_tr, max_query_len_te)

    X_tr, Q_tr=pad_data(X_tr,Q_tr,max_story_len, max_query_len)
    X_te, Q_te=pad_data(X_te,Q_te,max_story_len, max_query_len)

    input_facts = Input((max_story_len,))
    question = Input((max_query_len,))

    # A
    A= Embedding(input_dim=vocab_size,
                              output_dim=64)
    # C
    C=Embedding(input_dim=vocab_size,
                              output_dim=max_query_len)
    # B
    B=Embedding(input_dim=vocab_size,
                               output_dim=64,
                               input_length=max_query_len)

    input_A = A(input_facts)
    input_C = C(input_facts)
    question_B = B(question)

    input_question_match = dot([input_A, question_B], axes=(2, 2))
    Probs = Activation('softmax')(input_question_match)


    size=keras.backend.int_shape(input_C)[2]

    # Start of loop
    max_hops=2 
    if max_hops==0: 
        O = add([Probs, input_C])
    for i in range(max_hops): 
        input_C=Dense(size)(input_C)  
        O = add([Probs, input_C]) 
        input_question_match = dot([input_A, question_B], axes=(2, 2)) 
        input_question_match = add([input_question_match,O]) 
        Probs = Activation('softmax')(input_question_match) 
    # End of loop

    O = Permute((2, 1))(O)
    final_match = concatenate([O, question_B])
    size=keras.backend.int_shape(final_match)[2]
    weights = Dense(size, activation='softmax')(final_match)
    merged=merge([final_match, weights], mode='mul')
    answer=Flatten()(merged)
    answer=Dropout(0.3)(answer) # ADDED 25.03
    answer = Dense(vocab_size)(answer)  # (samples, vocab_size)
    answer = Activation('softmax')(answer)
    model = Model([input_facts, question], answer)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])

    model.summary()

    return X_tr,Q_tr,y_tr,X_te,Q_te,y_te,model
