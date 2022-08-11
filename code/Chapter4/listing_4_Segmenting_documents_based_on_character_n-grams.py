def segmentDocumentCharNgrams(filename, nb_words_per_segment, ngram_size):
    wordsDict={} <1>
    words=[]
    with open(filename, "r") as f:
         for line in f:
            line=line.rstrip().replace(' ','#') <2>
            char_ngrams_list=ngrams(list(line),ngram_size) <3>
            for char_ngram in char_ngrams_list:
                joined=''.join(char_ngram))
                words.append(joined) <4>
                wordsDict[joined]=1 <5>
    f.close()
    segments=[words[i:i+nb_words_per_segment] for i in xrange(0,len(words),
    [CA]nb_words_per_segment)] <6>
    return segments, wordsDict <7>
