embedding = Embedding(len(vocab) + 1,
                      embedding_dimension,
                      weights=[embedding_matrix],
                      input_length=max_len,
                      trainable=False)
