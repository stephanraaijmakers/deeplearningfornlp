def create_tokenizer(trainingdata, testdata):
    f=open(trainingdata, "r")
    text=[]

    for line in f:
        m=re.match("^\d+\s([^\.]+)[\.].*",line.rstrip()) 
        if m:
            text.append(m.group(1))
        else:
            m=re.match("^\d+\s([^\?]+)[\?]\s\t([^\t]+)",
                line.rstrip()) 
            if m:
                text.append(m.group(1)+' '+m.group(2))
    f.close()

    f=open(testdata, "r")
    for line in f:
        m=re.match("^\d+\s([^\.]+)[\.].*",line.rstrip()) 
        if m:
            text.append(m.group(1))
        else:
            m=re.match("^\d+\s([^\?]+)[\?].*",line.rstrip()) 
            if m:
                text.append(m.group(1))
    f.close()

    vocabulary=set([word for word in text]) 
    max_words = len(vocabulary)
    tokenizer = Tokenizer(
    num_words=max_words, char_level=False, split=' ') 
    tokenizer.fit_on_texts(text) 
    return tokenizer, max_words
