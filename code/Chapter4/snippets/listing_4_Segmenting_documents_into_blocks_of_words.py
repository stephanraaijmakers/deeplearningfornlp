def segmentDocumentWords(filename, nb_words_per_segment):
    wordsDict={}
    words=[]
    with open(filename, "r") as f:
         for line in f:
            tokens=line.rstrip().split(" ") <1>
            for token in tokens:
                if token!='':
                    words.append(token) <2>
                    wordsDict[token]=1 <3>

    f.close()
    segments=[words[i:i+nb_words_per_segment] for i in xrange(
    [CA]0,len(words),nb_words_per_segment)] <4>
    return segments, len(wordsDict) <5>
