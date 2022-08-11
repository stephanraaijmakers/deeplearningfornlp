def babify_dimin(fname):
    f=open(fname,"r")
    for line in f:
        features=line.rstrip().split(",")
        label=features.pop()
        fA=[] 
        for feature in features: 
            if feature=="=":
                feature="eq"
            elif feature =="-":
                feature="dash"
            elif feature=="+":
                feature="plus"
            elif feature=="@":
                feature="schwa"
            elif feature=='{':
                feature="lbr"
            elif feature=='}':
                feature="rbr"
            fA.append(feature)
        print "1 %s."%(' '.join(fA)) 
        print "2 suffix %s? \t%s\t%s"%(fA[-1],label,"1") 
     f.close()
