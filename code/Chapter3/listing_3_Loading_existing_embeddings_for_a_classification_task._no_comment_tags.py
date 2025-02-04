def load_embedding(f, vocab, embedding_dimension):
    embedding_index = {}
    f = open(f)
    n=0
    for line in f:
        values = line.split()
        word = values[0]
        if word in vocab: #only store words in current vocabulary
            coefs = np.asarray(values[1:], dtype='float32')
            if n: #skip header line
                embedding_index[word] = coefs
            n+=1
    f.close()

    embedding_matrix = np.zeros((len(vocab) + 1, embedding_dimension))
    for word, i in vocab.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
