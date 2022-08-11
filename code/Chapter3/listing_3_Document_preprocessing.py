docs=["Chuck Berry rolled over everyone who came before him ? and turned up
    [CA]everyone who came after. We'll miss you",
      "Help protect the progress we've made in helping millions of
    [CA]Americans get covered.",
      "Let's leave our children and grandchildren a planet that's healthier
    [CA]than the one we have today.",
      "The American people are waiting for Senate leaders to do their jobs.",
      "We must take bold steps now ? climate change is already impacting
    [CA]millions of people.",
      "Don't forget to watch Larry King tonight",
      "Ivanka is now on Twitter - You can follow her",
      "Last night Melania and I attended the Skating with the Stars Gala at
    [CA]Wollman Rink in Central Park",
      "People who have the ability to work should. But with the government
    [CA]happy to send checks",
      "I will be signing copies of my new book"
      ]

docs=[d.lower() for d in docs] <1>

count_vect = CountVectorizer().fit(docs) <2>
tokenizer = count_vect.build_tokenizer() <3>

input_array=[] <4>
for doc in docs:
    x=[]
    for token in tokenizer(doc):
        x.append(count_vect.vocabulary_.get(token))
    input_array.append(x)

max_len=max([len(d) for d in input_array]) <5>

input_array=pad_sequences(input_array, maxlen=max_len,
[CA]padding='post') <6>
