def splitData(X,y, max_samples_per_author=10):
    X,y=shuffle(X,y,random_state=42)
    AuthorsX={}

    for (x,y) in zip(X,y): <1>
            y=np.where(y==1)[0][0]
            if y in AuthorsX:
                    AuthorsX[y].append(x)
            else:
                    AuthorsX[y]=[x]

    X_left=[]
    X_right=[]
    y_lr=[]

    Done={}
    for author in AuthorsX:
        nb_texts=len(AuthorsX[author])
        nb_samples=min(nb_texts, max_samples_per_author)
        left_docs=np.array(AuthorsX[author])
        random_indexes=np.random.choice(left_docs.shape[0], nb_samples,
        [CA]replace=False)
        left_sample=np.array(AuthorsX[author])[random_indexes]
        for other_author in AuthorsX: <2>
            if  (other_author,author) in Done:
                    pass
            Done[(author,other_author)]=1

            right_docs=np.array(AuthorsX[other_author])

            nb_samples_other=min(len(AuthorsX[other_author]),
            [CA]max_samples_per_author)
            random_indexes_other=np.random.choice(right_docs.shape[0],
            [CA]nb_samples_other, replace=False)
            right_sample=right_docs[random_indexes_other]

            for (l,r) in zip(left_sample,right_sample): <3>
                    X_left.append(l)
                    X_right.append(r)
                    if author==other_author:
                            y_lr.append(1.0)
                    else:
                            y_lr.append(0.0)
    return np.array(X_left),np.array(X_right),np.array(y_lr) <4>
