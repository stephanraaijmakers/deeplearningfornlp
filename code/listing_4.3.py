def vectorizeDocumentsBOW(path, labelDict, nb_words_per_segment):
    files = [filename for filename in listdir(path) if isfile(join(path, filename))]
    segments=[]
    labels=[]
    globalDict={} 

    for file in files:
        match=re.match("^.*12[A-Z][a-z]+([A-Z]+).+",file)
        if match:
            label=ord(match.group(1))-65 
        else:
            print('Skipping filename:%s'%(file))
            continue

        (segmented_document,wordDict)=segmentDocumentWords(join(path,file),nb_words_per_segment) 

        globalDict=mergeDictionaries(globalDict,wordDict) 

        segments.extend(segmented_document) 

        for segment in segmented_document:
            labels.append(label)

    vocab_len=len(globalDict) 

    labels=[labelDict[x] for x in labels]
    nb_classes=len(labelDict)

    X=[]
    y=[]

    for segment in segments:
        segment=' '.join(segment)
        X.append(pad_sequences([hashing_trick(segment, round(vocab_len*1.3))], nb_words_per_segment)[0]) 

    y=np_utils.to_categorical(labels, nb_classes)

    return np.array(X), y, vocab_len 
