def createData():
    max_words = 1000
    maxlen=100 #500
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words) 

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen, padding='post') 
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen,padding='post')

    num_classes=max(y_train)+1

    word_index = imdb.get_word_index(path="imdb_word_index.json") 

    # Inverse index: number=>word
    RevWordIndex = {}
    RevWordIndex[0]="" #n/a"
    for key, value in word_index.items():
        RevWordIndex[value]=key

    return x_train, y_train, x_test, y_test, RevWordIndex, num_classes 
