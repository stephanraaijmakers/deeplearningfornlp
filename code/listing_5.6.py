
def process_stories_n_context(filename,tokenizer,vocab_size,use_context=0): 
    f=open(filename,"r")
    X=[]
    Q=[]
    y=[]
    max_story_len=0
    max_query_len=0

    for line in f:
        m=re.match("^(\d+)\s(.+)\.",line.rstrip()) 
        if m:
            if int(m.group(1))==1:
                story={}
            story[int(m.group(1))]=m.group(2)
        else:
            m=re.match("^\d+\s(.+)\?\s\t([^\t]+)\t(.+)",line.rstrip()) 
            if m:
                question=m.group(1)
                answer=m.group(2)
                answer_ids=[int(x) for x in m.group(3).split(" ")]
                facts=' '.join([story[id] for id in answer_ids])
                all_facts=' '.join([story[id] for id in story])
                facts_v=vectorize(facts,tokenizer)
                all_facts_v=vectorize(all_facts,tokenizer)

                if use_context==0: 
                    vectorized_fact=facts_v
                elif use_context==-1:
                    vectorized_fact=all_facts_v
                else:
                    x=min(use_context, len(story))
                    facts=' '.join([story[id] for id in answer_ids])+' '
                    n=0
                    for id in story:
                        if n<x and id not in answer_ids:
                            facts+=story[id]+' '
                            n+=1
                    vectorized_fact=vectorize(facts,tokenizer)
                l=len(vectorized_fact)
                if l>max_story_len:
                    max_story_len=l
                vectorized_question=vectorize(question,tokenizer) 
                l=len(vectorized_question)
                if l>max_query_len:
                    max_query_len=l

                vectorized_answer=vectorize(answer,tokenizer) 

                X.append(vectorized_fact) 
                Q.append(vectorized_question)
                answer=np.zeros(vocab_size)
                answer[vectorized_answer[0]]=1
                y.append(answer)
    f.close()

    return np.array(X),np.array(Q),np.array(y), max_story_len, max_query_len

