def createLabelDictOneFile(path):
    files = [join(path,filename) for filename in listdir(path) if isfile(join(path, filename))]
    labelDict={}

    for file in files:
        match=re.match("^.*\/?12[A-Z][a-z]+([A-Z]+).+",file) 
        if match:
            label=ord(match.group(1))-65 
        else:
            print('Skipping filename:%s'%(file))
            continue
        if label not in labelDict:
            labelDict[label]=len(labelDict) 

    return labelDict 

