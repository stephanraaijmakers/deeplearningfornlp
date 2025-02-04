def process_stories(filename,tokenizer,max_story_len,max_query_len,
[CA]vocab_size,use_context=False): <1>
    f=open(filename,"r")
    X=[]
    Q=[]
    y=[]
    n_questions=0

    for line in f:
        m=re.match("^(\d+)\s(.+)\.",line.rstrip()) <2>
        if m:
            if int(m.group(1))==1: <3>
                story={}
            story[int(m.group(1))]=m.group(2) <4>
        else:
            m=re.match("^\d+\s(.+)\?\s\t([^\t]+)\t(.+)",
                line.rstrip()) <5>
            if m:
                question=m.group(1)
                answer=m.group(2)
                answer_ids=[int(x) for x in m.group(3).split(" ")] <6>
                if use_context==False: <7>
                    facts=' '.join([story[id] for id in answer_ids])
                    vectorized_fact=vectorize(facts,tokenizer)
                else: <8>
                    vectorized_fact=vectorize(' '.join(story.values()),
                    [CA]tokenizer)
                vectorized_question=
                    vectorize(question,tokenizer) <9>
                vectorized_answer=
                    vectorize(answer,tokenizer) <10>

                X.append(vectorized_fact) <11>

                Q.append(vectorized_question) <12>

                answer=np.zeros(vocab_size) <13>
                answer[vectorized_answer[0]]=1
                y.append(answer)
    f.close()

    X=pad_sequences(X,maxlen=max_story_len)<14>
    Q=pad_sequences(Q,maxlen=max_query_len)

    return np.array(X),np.array(Q),np.array(y) <15>
