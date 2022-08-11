def process_data(textFile,window_size):
    docs = getLines(textFile)
    vocab = dict()
    create_vocabulary(vocab, docs)
    docid=0
    contexts=[]
    docids=[]
    targets=[]

    f=open("docs.legenda","w")
    for s in docs:
        f.write("%d %s\n"%(docid,s))
        docids.append(docid)
        ngs=list(ngrams(s.split(), window_size))
        for i in range(len(ngs)-1):
            cs=[docid]
            ng=ngs[i]
            for w in ng:
                w=re.sub("[.,:;'\"!?()]+","",w.lower())
                cs.append(vocab[w])
            contexts.append(cs)
            target_word=re.sub("[.,:;'\"!?()]+","",ngs[i+1][0].lower())
            targets.append(vocab[target_word])
        docid+=1
    f.close()
    return np.array(contexts),np.array(docids),np.array(targets),vocab

