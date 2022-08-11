
def create_model(trainingData, testData, context=False):
    tokenizer,vocab_size, max_story_len, max_query_len=create_tokenizer(trainingData,testData) 

    X_tr,Q_tr,y_tr=process_stories(trainingData,tokenizer,max_story_len, max_query_len,vocab_size,use_context=context) 

    X_te,Q_te,y_te=process_stories(testData,tokenizer,max_story_len, max_query_len,vocab_size,use_context=context) 

    embedding=layers.Embedding(vocab_size,100) 

    story = layers.Input(shape=(max_story_len,), dtype='int32') 
    encoded_story = embedding(story) 
    encoded_story = SimpleRNN(30)(encoded_story) 

    question = layers.Input(shape=(max_query_len,), dtype='int32') 
    encoded_question = embedding(question) 
    encoded_question = SimpleRNN(30)(encoded_question) 

    merged = layers.concatenate([encoded_story, encoded_question]) 
    preds = layers.Dense(vocab_size, activation='softmax')(merged) 

    model = Model([story, question], preds) 
    model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

    return X_tr,Q_tr,y_tr,X_te,Q_te,y_te,model

