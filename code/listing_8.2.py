
def vectorizeString(s,lexicon):
    vocabSize = len(lexicon)
    result = one_hot(s,round(vocabSize*1.5))
    return result
