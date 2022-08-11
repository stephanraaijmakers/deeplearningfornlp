def segmentDocumentCharNgrams(filename, nb_words_per_segment, ngram_size):
    wordsDict={} 
    words=[]
    with open(filename, "r") as f:
         for line in f:
            line=line.rstrip().replace(' ','#') 
            char_ngrams_list=ngrams(list(line),ngram_size) 
            for char_ngram in char_ngrams_list:
                joined=''.join(char_ngram))
                words.append(joined) 
                wordsDict[joined]=1 
    f.close()
    segments=[words[i:i+nb_words_per_segment] for i in xrange(0,len(words),nb_words_per_segment)] 
    return segments, wordsDict 
