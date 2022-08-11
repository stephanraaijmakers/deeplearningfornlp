Stored={}
for x in range(45):
    for y in range(45):
        if x==y:
            continue
        if (x,y) not in Stored and (y,x) not in Stored:
            Stored[(x,y)]=1

