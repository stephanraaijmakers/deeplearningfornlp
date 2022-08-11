def create_tokenizer(trainingdata, testdata):
    f=open(trainingdata, "r")
    text=[]

    for line in f:
        m=re.match("^\d+\s([^\.]+)[\.].*",line.rstrip()) <1>
        if m:
            text.append(m.group(1))
        else:
            m=re.match("^\d+\s([^\?]+)[\?]\s\t([^\t]+)",
                line.rstrip()) <2>
            if m:
                text.append(m.group(1)+' '+m.group(2))
    f.close()

    f=open(testdata, "r")
    for line in f:
        m=re.match("^\d+\s([^\.]+)[\.].*",line.rstrip()) <3>
        if m:
            text.append(m.group(1))
        else:
            m=re.match("^\d+\s([^\?]+)[\?].*",line.rstrip()) <4>
            if m:
                text.append(m.group(1))
    f.close()

    vocabulary=set([word for word in text]) <5>
    max_words = len(vocabulary)
    tokenizer = Tokenizer(
    [CA]num_words=max_words, char_level=False, split=' ') <6>
    tokenizer.fit_on_texts(text) <7>
    return tokenizer, max_words
