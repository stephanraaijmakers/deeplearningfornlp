def babify_pp(fname):
    inp=open(fname,"r")
    for line in inp:
        m=re.match("^(.+),([^,]+)$",line.rstrip())
        if m:
            features=m.group(1).split(",")
            label=m.group(2)
            n=1
            print "1 %s V." %(features[0])
            print "2 %s N." %(features[1])
            pp_str=features[2] +' ' + features[3]
            print "%d attach %s? \t%s\t%s" %
 (n,pp_str,label, ' '.join([str(x) for x in range(1,3)]))
    inp.close()
