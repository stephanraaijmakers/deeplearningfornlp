def babify_conll02(fname):
    f=open(fname,"r")
    Lex={} 
    for line in f:
        if re.match(".+DOCSTART.+",line):
            continue
        m=re.match("^([^\s]+)\s+([^\s]+)\s+(.+)$",line.rstrip()) 
        if m:
            word=m.group(1)
            pos=m.group(2)
            if word in Lex:
                if pos not in Lex[word]: 
                    Lex[word].append(pos)
            else:
                Lex[word]=[pos]
    f.seek(0) 

    ngramsize=3 
    focus=1 
    story=""
    for line in f:
        if re.match(".+DOCSTART.+",line):
            continue
        if re.match("^\s*$",line.rstrip()): 
            ngrs=ngrams(story,ngramsize)
            n=1
            ambig=False
            for ngr in ngrs:
                fact="%d"%(n) 
                i=0
                for w in ngr:
                    word_plus_pos=w.split("#")
                    word=word_plus_pos[0]
                    pos=word_plus_pos[1]
                    lex_pos='_'.join(Lex[word])
                    if i==focus: 
                        fact+=" %s"%(lex_pos)
                        if '_' in lex_pos:
                            ambig=True
                            unique_pos=pos
                            ambig_word=word
                            ambig_pos=lex_pos
                    elif i==ngramsize-1: 
                        fact+=" %s."%(lex_pos)
                        print fact
                    else:
                        fact+=" %s"%(lex_pos)
                    i+=1
                if ambig: 
                    n+=1
                    ambig=False
                    if n>2:
                        print "%d pos %s? \t%s\t%d %d"%(n,ambig_pos,
[CA]unique_pos,n-2,n-1)
                    else:
                        print "%d pos %s? \t%s\t%d"%(n,ambig_pos,
[CA]unique_pos,n-1)
                    n=0

                n+=1
            story=""
        else:
            m=re.match("^([^\s]+)\s+([^\s]+)\s+(.+)$",line.rstrip())
            if m:
                story+=m.group(1)+"#"+m.group(2)+" "
    f.close()
    exit(0)

