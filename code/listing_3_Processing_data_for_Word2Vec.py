def process_data(textFile,window_size):
    couples=[]
    labels=[]
    sentences = getLines(textFile)
    vocab = dict()
    create_vocabulary(vocab, sentences)
    vocab_size=len(vocab)
    for s in sentences:
        words=[]
        for w in s.split(" "):
            w=re.sub("[.,:;'\"!?()]+","",w.lower())
            if w!='':
                words.append(vocab[w])
        c,l=skipgrams(words,vocab_size,window_size=window_size)
        couples.extend(c)
        labels.extend(l)
    return vocab,couples,labels
