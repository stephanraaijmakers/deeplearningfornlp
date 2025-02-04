docs=["Chuck Berry rolled over everyone who came before him ? and turned up
    everyone who came after. We'll miss you",
      "Help protect the progress we've made in helping millions of
    Americans get covered.",
      "Let's leave our children and grandchildren a planet that's healthier
    than the one we have today.",
      "The American people are waiting for Senate leaders to do their jobs.",
      "We must take bold steps now ? climate change is already impacting
    millions of people.",
      "Don't forget to watch Larry King tonight",
      "Ivanka is now on Twitter - You can follow her",
      "Last night Melania and I attended the Skating with the Stars Gala at
    Wollman Rink in Central Park",
      "People who have the ability to work should. But with the government
    happy to send checks",
      "I will be signing copies of my new book"
      ]

docs=[d.lower() for d in docs] 

count_vect = CountVectorizer().fit(docs) 
tokenizer = count_vect.build_tokenizer() 

input_array=[] 
for doc in docs:
    x=[]
    for token in tokenizer(doc):
        x.append(count_vect.vocabulary_.get(token))
    input_array.append(x)

max_len=max([len(d) for d in input_array]) 

input_array=pad_sequences(input_array, maxlen=max_len,
padding='post') 
