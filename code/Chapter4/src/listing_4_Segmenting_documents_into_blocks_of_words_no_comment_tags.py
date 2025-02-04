def segmentDocumentWords(filename, nb_words_per_segment):
    wordsDict={}
    words=[]
    with open(filename, "r") as f:
         for line in f:
            tokens=line.rstrip().split(" ") 
            for token in tokens:
                if token!='':
                    words.append(token) 
                    wordsDict[token]=1 

    f.close()
    segments=[words[i:i+nb_words_per_segment] for i in xrange(0,len(words),nb_words_per_segment)] 
    return segments, len(wordsDict) 
