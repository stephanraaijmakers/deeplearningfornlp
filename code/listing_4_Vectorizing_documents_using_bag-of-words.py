def vectorizeDocumentsBOW(path, labelDict, nb_words_per_segment):
    files = [filename for filename in listdir(path) if isfile(
    [CA]join(path, filename))]
    segments=[]
    labels=[]
    globalDict={} <1>

    for file in files:
        match=re.match("^.*12[A-Z][a-z]+([A-Z]+).+",file)
        if match:
            label=ord(match.group(1))-65 <2>
        else:
            print('Skipping filename:%s'%(file))
            continue

        (segmented_document,wordDict)=segmentDocumentWords(join(path,file),
        [CA]nb_words_per_segment) <3>

        globalDict=mergeDictionaries(globalDict,wordDict) <4>

        segments.extend(segmented_document) <5>

        for segment in segmented_document:
            labels.append(label)

    vocab_len=len(globalDict) <6>

    labels=[labelDict[x] for x in labels]
    nb_classes=len(labelDict)

    X=[]
    y=[]

    for segment in segments:
        segment=' '.join(segment)
        X.append(pad_sequences([hashing_trick(
        [CA]segment, round(vocab_len*1.3))],
        [CA]nb_words_per_segment)[0]) <7>

    y=np_utils.to_categorical(labels, nb_classes)

    return np.array(X), y, vocab_len <8>
