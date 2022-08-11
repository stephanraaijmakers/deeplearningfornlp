embedding_matrix=load_embedding(sys.argv[3],vocab,
[CA]embedding_dimension) <1>
embedding = Embedding(len(vocab) + 1,
                            embedding_dimension,
                            weights=[embedding_matrix],
                            input_length=max_len,
                            trainable=False) <2>
