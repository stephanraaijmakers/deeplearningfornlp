input_w1 = Input((1,))
input_w2 = Input((1,))
input_w3 = Input((1,))
input_docid=Input((1,))

# Embeddings
embedding = Embedding(vocab_size, vector_dim, input_length=1, name='embedding')

vector_dim_doc=vector_dim
embedding_doc=Embedding(len(docids)+1,vector_dim_doc)

docid=embedding_doc(input_docid)
docid = Reshape((vector_dim_doc,1))(docid)

w1 = embedding(input_w1)
w1 = Reshape((vector_dim, 1))(w1)

w2 = embedding(input_w2)
w2 = Reshape((vector_dim, 1))(w2)

w3 = embedding(input_w3)
w3 = Reshape((vector_dim, 1))(w3)

context_docid=concatenate([w1,w2,w3,docid])
context_docid=Flatten()(context_docid)
output = Dense(vocab_size,activation='softmax')(context_docid)
model = Model(input=[input_w1, input_w2, input_w3, input_docid], output=output)
model.compile(loss='sparse_categorical_crossentropy',
[CA]optimizer='adam',metrics=['acc'])

print model.summary()

epochs=int(sys.argv[2])

model.fit_generator(generator(contexts,targets,100),
[CA]steps_per_epoch=100, epochs=epochs)
