def loadData(train, test):

  global Lexicon

  with io.open(train,encoding = "ISO-8859-1") as f:
      trainD = f.readlines() 
  f.close()

  with io.open(test,encoding = "ISO-8859-1") as f:
      testD = f.readlines() 
  f.close()

  all_text=[]
  for line in trainD:
      m=re.match("^(.+),[^\s]+$",line)
      if m:
        all_text.extend(m.group(1).split(" ")) 

    for line in testD:
      m=re.match("^(.+),[^\s]+$",line)
      if m:
        all_text.extend(m.group(1).split(" ")) 

  Lexicon=set(all_text) 

  x_train=[]
  y_train=[]
  x_test=[]
  y_test=[]

  for line in trainD: 
      m=re.match("^(.+),([^\s]+)$",line)
      if m:
        x_train.append(vectorizeString(m.group(1),Lexicon))
        y_train.append(processLabel(m.group(2)))

  for line in testD: 
      m=re.match("^(.+),([^\s]+)$",line)
      if m:
        x_test.append(vectorizeString(m.group(1),Lexicon))
        y_test.append(processLabel(m.group(2)))

  return (np.array(x_train),np.array(y_train)),(np.array(x_test),
  np.array(y_test)) 
