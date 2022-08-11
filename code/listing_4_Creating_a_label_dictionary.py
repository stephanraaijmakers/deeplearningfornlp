def createLabelDict(pathTraining, pathTest):
    filesTraining = [join(pathTraining,filename) for filename in
    [CA]listdir(pathTraining) if isfile(join(pathTraining, filename))]
    filesTest = [join(pathTest,filename) for filename in listdir(pathTest)
    [CA]if isfile(join(pathTest, filename))]
    files=filesTraining+filesTest

    labelDict={}

    for file in files:
        match=re.match("^.*\/?12[A-Z][a-z]+([A-Z]+).+",file) <1>
        if match:
            label=ord(match.group(1))-65 <2>
        else:
            print('Skipping filename:%s'%(file))
            continue
        if label not in labelDict:
            labelDict[label]=len(labelDict) <3>
    return labelDict <4>
