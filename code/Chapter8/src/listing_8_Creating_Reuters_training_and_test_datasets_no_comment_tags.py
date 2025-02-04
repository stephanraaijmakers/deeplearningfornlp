Tried={} 
for (topic1,topic2) in Stored:
    for (topic3,topic4) in Stored:
        if (topic1,topic2)==(topic3,topic4): 
            continue
        if topic1 in (topic3,topic4) or topic2 in (topic3,topic4):
            continue
        if (topic1,topic2) in Tried or (topic3,topic4) in Tried:
            continue
        Tried[(topic1,topic2)]=1
        Tried[(topic3,topic4)]=1

        ClassLexicon={} 
        ClassLexicon[topic1]=ClassLexicon[topic2]=0
        ClassLexicon[topic3]=ClassLexicon[topic4]=1


        indices_train1=[i for i in range(len(y_train)) if y_train[i] in [topic1,topic2]] 
        indices_test1=[i for i in range(len(y_test)) if cy_test[i] in [topic1,topic2]]
        indices_train2=[i for i in range(len(y_train)) if y_train[i] in [topic3,topic4]]
        indices_test2=[i for i in range(len(y_test)) if y_test[i] in [topic3,topic4]]

        x1_train=np.array([x_train[i] for i in indices_train1])
        y1_train=np.array([processLabel(y_train[i]) for i in indices_train1])

        ClassLexicon={} 

        x1_test=np.array([x_test[i] for i in indices_test1])
        y1_test=np.array([processLabel(y_test[i]) for i in indices_test1])

        ClassLexicon={} 

        x2_train=np.array([x_train[i] for i in indices_train2])
        y2_train=np.array([processLabel(y_train[i]) for i in indices_train2])

        ClassLexicon={}

        x2_test=np.array([x_test[i] for i in indices_test2])
        y2_test=np.array([processLabel(y_test[i]) for i in indices_test2])

        num_classes1=2
        num_classes2=2
        max_length=1000

        x1_train = pad_sequences(x1_train, maxlen=max_length,
        padding='post') 
        y1_train = keras.utils.to_categorical(y1_train, num_classes1)
        x1_test = pad_sequences(x1_test, maxlen=max_length, padding='post')
        y1_test = keras.utils.to_categorical(y1_test, num_classes1)
        x2_train = pad_sequences(x2_train, maxlen=max_length, padding='post')
        y2_train = keras.utils.to_categorical(y2_train, num_classes2)
        x2_test = pad_sequences(x2_test, maxlen=max_length, padding='post')
        y2_test = keras.utils.to_categorical(y2_test, num_classes2)

        if len(x1_train)<300 or len(x2_train)<300: 
            continue

        min_train=min(len(x1_train),len(x2_train)) 
        x1_train=x1_train[:min_train]
        x2_train=x2_train[:min_train]
        y1_train=y1_train[:min_train]
        y2_train=y2_train[:min_train]

        min_test=min(len(x1_test),len(x2_test))
        x1_test=x1_test[:min_test]
        x2_test=x2_test[:min_test]
        y1_test=y1_test[:min_test]
        y2_test=y2_test[:min_test]
