def create_model(trainingData, testData, context):

    tokenizer,vocab_size=create_tokenizer(trainingData,testData)

    X_tr,Q_tr,y_tr,max_story_len_tr, max_query_len_tr=
        process_stories_n_context(trainingData,tokenizer,vocab_size,
        use_context=context)
    X_te,Q_te,y_te, max_story_len_te, max_query_len_te=
        process_stories_n_context(testData,tokenizer,vocab_size,
        use_context=context)

    max_story_len=max(max_story_len_tr, max_story_len_te)
    max_query_len=max(max_query_len_tr, max_query_len_te)

    X_tr, Q_tr=pad_data(X_tr,Q_tr,max_story_len, max_query_len)
    X_te, Q_te=pad_data(X_te,Q_te,max_story_len, max_query_len)

    (...)
