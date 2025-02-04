def vectorize(s, tokenizer):
    vector=tokenizer.texts_to_sequences([s])
    return vector[0]
