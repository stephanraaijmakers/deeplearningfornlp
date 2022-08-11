from scipy.spatial.distance import cosine 
from textblob import TextBlob


def readOutBertModel(tokenizer, functionBert, textCSV): 

    examples=convert_text_to_examples(textCSV) 
    features=convert_examples_to_features(tokenizer,examples)

    pred=functionBert(features) 

    tags=TextBlob(examples[0].text).tags 

    vectors=[]
    text=""
    for i in range(nb_words): 
        if tags[i][1] in ['NNP','NN','NNS']:
            vectors.append(pred[0][0][i])
            text+=words[i]+" "

    # Words: money bank bank account river bank

    same_bank = 1 - cosine(vectors[1], vectors[2]) 
    other_bank=1 - cosine(vectors[1], vectors[5]) 

    print('Vector similarity for  *similar*  meanings:  %.2f' % same_bank) 
    print('Vector similarity for *different* meanings:  %.2f' % other_bank)

    return (vectors, text) 
