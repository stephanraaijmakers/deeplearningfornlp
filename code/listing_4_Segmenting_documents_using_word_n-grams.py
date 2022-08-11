from nltk.util import ngrams

def segmentDocumentNgrams(filename, nb_words_per_segment, ngram_size):
    wordsDict={}
    words=[]
    with open(filename, "r") as f:
         for line in f:
            ngrams_list=ngrams(line.rstrip(),ngram_size) <1>
            for ngram in ngram_list:
                joined='_'.join(ngram) 
                words.append(joined) 
                wordsDict[joined]=1 <4>
    f.close()
    segments=[words[i:i+nb_words_per_segment] for i in xrange(0,len(words),
    [CA]nb_words_per_segment)] <5>
    return segments, wordsDict 
