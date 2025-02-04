from nltk.util import ngrams

def segmentDocumentNgrams(filename, nb_words_per_segment, ngram_size):
    wordsDict={}
    words=[]
    with open(filename, "r") as f:
         for line in f:
            ngrams_list=ngrams(line.rstrip(),ngram_size) 
            for ngram in ngram_list:
                joined='_'.join(ngram)
                words.append(joined)
                wordsDict[joined]=1 
    f.close()
    segments=[words[i:i+nb_words_per_segment] for i in xrange(0,len(words),
    nb_words_per_segment)] 
    return segments, wordsDict
