def processLabel(x):
    if x in ClassLexicon:
        return ClassLexicon[x]
    else:
        ClassLexicon[x]=len(ClassLexicon)
        return ClassLexicon[x]
