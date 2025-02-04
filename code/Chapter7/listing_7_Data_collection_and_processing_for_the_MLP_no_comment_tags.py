def createData(stopwords, filterStopwords=False):

    (x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=None, test_split=0.2)  

    num_classes=max(y_train)+1

    word_index = reuters.get_word_index(path="reuters_word_index.json")

    # Inverse index: number=>word
    RevWordIndex = {} 
    for key, value in word_index.items():
        RevWordIndex[value]=key

    max_words = 10000

    tokenizer = Tokenizer(num_words=max_words) 
    x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
    x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    if filterStopwords: 
      j=0
      for x in x_train:
          n=1
          for w in x:
              if RevWordIndex[n] in stopwords:
                  x_train[j][n-1]=0.0
              n+=1
          j+=1

      j=0
      for x in x_test:
          n=1
          for w in x:
              if RevWordIndex[n] in stopwords:
                  x_test[j][n-1]=0.0
              n+=1
          j+=1

    return x_train, y_train, x_test, y_test, RevWordIndex,
num_classes 
