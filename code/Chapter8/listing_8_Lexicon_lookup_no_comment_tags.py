global ClassLex
ClassLex={}

def lookup(feat):
    if feat not in Lex:
        Lex[feat]=len(Lex)
    return Lex[feat]

def class_lookup(feat):
    if feat not in ClassLex:
        ClassLex[feat]=len(ClassLex)
    return ClassLex[feat]
