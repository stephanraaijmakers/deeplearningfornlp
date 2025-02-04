def vectorize(s, tokenizer):
    vector=tokenizer.texts_to_sequences([s])
    if vector[0]!='':
        return vector[0]
