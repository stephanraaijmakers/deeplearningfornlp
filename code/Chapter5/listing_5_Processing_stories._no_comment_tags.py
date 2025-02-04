def process_stories(filename,tokenizer,max_story_len,max_query_len,
vocab_size,use_context=False): 
    f=open(filename,"r")
    X=[]
    Q=[]
    y=[]
    n_questions=0

    for line in f:
        m=re.match("^(\d+)\s(.+)\.",line.rstrip()) 
        if m:
            if int(m.group(1))==1: 
                story={}
            story[int(m.group(1))]=m.group(2) 
        else:
            m=re.match("^\d+\s(.+)\?\s\t([^\t]+)\t(.+)",
                line.rstrip()) 
            if m:
                question=m.group(1)
                answer=m.group(2)
                answer_ids=[int(x) for x in m.group(3).split(" ")] 
                if use_context==False: 
                    facts=' '.join([story[id] for id in answer_ids])
                    vectorized_fact=vectorize(facts,tokenizer)
                else: 
                    vectorized_fact=vectorize(' '.join(story.values()),
                    tokenizer)
                vectorized_question=
                    vectorize(question,tokenizer) 
                vectorized_answer=
                    vectorize(answer,tokenizer) 

                X.append(vectorized_fact) 

                Q.append(vectorized_question) 

                answer=np.zeros(vocab_size) 
                answer[vectorized_answer[0]]=1
                y.append(answer)
    f.close()

    X=pad_sequences(X,maxlen=max_story_len)
    Q=pad_sequences(Q,maxlen=max_query_len)

    return np.array(X),np.array(Q),np.array(y) 
