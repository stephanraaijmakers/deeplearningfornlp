def load_conll(train, test):
    x1_train=[]
    y1_train=[]
    x1_test=[]
    y1_test=[]
    x2_train=[]
    y2_train=[]
    x2_test=[]
    y2_test=[]

    tr=open(train,"r") <1>
    for line in tr:
        if line.rstrip()=='':
            continue
        features=line.rstrip().split("|") <1>
        target=features.pop().split(" ")
        target_word=target[0]
        target_y1=target[1]
        target_y2=target[2]

        y1_train.append(class_lookup(target_y1))
        y2_train.append(class_lookup(target_y2))

        l=lookup(target_word) <2>
        x1=[l]
        x2=[l]
        for feature in features:
            if feature=='':
                continue
            feature_split=feature.split(" ")
            x1.append(lookup(feature_split[0]))
            x1.append(lookup(feature_split[1]))
            x2.append(lookup(feature_split[0]))
            x2.append(lookup(feature_split[2]))
        x1_train.append(x1)
        x2_train.append(x2)
    tr.close()

    te=open(test,"r") <3>
    for line in te:
        if line.rstrip()=='':
            continue
        features=line.rstrip().split("|")
        target=features.pop().split(" ")
        target_word=target[0]
        target_y1=target[1]
        target_y2=target[2]

        y1_test.append(class_lookup(target_y1))
        y2_test.append(class_lookup(target_y2))

        l=lookup(target_word)
        x1=[l]
        x2=[l]

        for feature in features:
            if feature=='':
                continue
            feature_split=feature.split(" ")
            x1.append(lookup(feature_split[0]))
            x1.append(lookup(feature_split[1]))
            x2.append(lookup(feature_split[0]))
            x2.append(lookup(feature_split[2]))
        x1_test.append(x1)
        x2_test.append(x2)
    te.close()

    return (np.array(x1_train), np.array(y1_train)),(np.array(x2_train),
    [CA]np.array(y2_train)),(np.array(x1_test),np.array(y1_test)),
    [CA](np.array(x2_test),np.array(y2_test)) <4>
