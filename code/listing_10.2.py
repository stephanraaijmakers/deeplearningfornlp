
def readSentencePairs(fn): 
    with open(fn) as f:
        lines = f.readlines() 
    pairs=zip(lines, lines[1:]) 
    paired_sentences=[[a.rstrip().split(),b.rstrip().split()] for (a,b) in pairs] 
    tokenD = get_base_dict()  
    for pairs in paired_sentences: 
        for token in pairs[0] + pairs[1]:
            if token not in tokenD:
                tokenD[token] = len(tokenD)
    tokenL = list(tokenD.keys())  
    return (paired_sentences,tokenD,tokenL) 
