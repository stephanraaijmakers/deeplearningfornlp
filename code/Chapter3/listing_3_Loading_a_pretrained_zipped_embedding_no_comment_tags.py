def load_embedding_zipped(f, vocab, embedding_dimension):
    embedding_index = {}
    with zipfile.ZipFile(f) as z:
        with z.open("glove.6B.100d.txt") as f:
            n=0
            for line in f:
                if n:
                    values = line.split()
                    word = values[0]
                    if word in vocab:
                        coefs = np.asarray(values[1:], dtype='float32')
                        embedding_index[word] = coefs
                n+=1
    z.close()
    embedding_matrix = np.zeros((len(vocab) + 1, embedding_dimension))
    for word, i in vocab.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
