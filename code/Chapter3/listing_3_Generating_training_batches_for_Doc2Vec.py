def generator(contexts, targets, batch_size):
    w1 = np.zeros((batch_size, 1))
    w2 = np.zeros((batch_size, 1))
    w3 = np.zeros((batch_size, 1))
    docid = np.zeros((batch_size, 1))
    batch_targets = np.zeros((batch_size,1))

    while True:
        for i in range(batch_size):
            index= random.randint(0,len(targets)-1)
            batch_targets[i] = targets[index]
            docid[i]=contexts[index][0]
            w1[i] = contexts[index][1]
            w2[i] = contexts[index][2]
            w3[i] = contexts[index][3]
        yield [w1,w2,w3,docid], [batch_targets]
