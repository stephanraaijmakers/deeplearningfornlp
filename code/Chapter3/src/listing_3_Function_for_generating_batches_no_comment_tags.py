def generator(target,context, labels, batch_size):
    batch_target = np.zeros((batch_size, 1))
    batch_context = np.zeros((batch_size, 1))
    batch_labels = np.zeros((batch_size,1))

    while True:
        for i in range(batch_size):
            index= random.randint(0,len(target)-1)
            batch_target[i] = target[index]
            batch_context[i]=context[index]
            batch_labels[i] = labels[index]
        yield [batch_target,batch_context], [batch_labels]

