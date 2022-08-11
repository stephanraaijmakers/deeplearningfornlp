
def create_model(trainingData, testData, context=False):
    tokenizer,vocab_size, max_story_len, max_query_len=create_tokenizer(
    [CA]trainingData,testData) <1>

    X_tr,Q_tr,y_tr=process_stories(trainingData,tokenizer,max_story_len,
        max_query_len,vocab_size,use_context=context) <2>

    X_te,Q_te,y_te=process_stories(testData,tokenizer,max_story_len,
        max_query_len,vocab_size,use_context=context) <3>

    embedding=layers.Embedding(vocab_size,100) <4>

    story = layers.Input(shape=(max_story_len,),
        dtype='int32') <5>
    encoded_story = embedding(story) <6>
    encoded_story = SimpleRNN(30)(encoded_story) <7>

    question = layers.Input(shape=(max_query_len,),
        dtype='int32') <8>
    encoded_question = embedding(question) <9>
    encoded_question = SimpleRNN(30)(encoded_question) <10>

    merged = layers.concatenate([encoded_story,
        encoded_question]) <11>
    preds = layers.Dense(vocab_size, activation=
        'softmax')(merged) <12>

    model = Model([story, question], preds) <13>
    model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

    return X_tr,Q_tr,y_tr,X_te,Q_te,y_te,model
