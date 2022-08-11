def readSentencePairs(fn): <1>
    with open(fn) as f:
        lines = f.readlines() <2>

    pairs=zip(lines, lines[1:]) <3>
    paired_sentences=[[a.rstrip().split(),b.rstrip().split()]
    [CA]for (a,b) in pairs] <4>

    tokenD = get_base_dict()  <5>

    for pairs in paired_sentences: <6>
        for token in pairs[0] + pairs[1]:
            if token not in tokenD:
                tokenD[token] = len(tokenD)
    tokenL = list(tokenD.keys())  <7>
    return (paired_sentences,tokenD,tokenL) <8>
