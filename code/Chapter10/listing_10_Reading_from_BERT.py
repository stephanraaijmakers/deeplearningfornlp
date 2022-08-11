from scipy.spatial.distance import cosine <1>
from textblob import TextBlob

def readOutBertModel(tokenizer, functionBert, textCSV): <2>

    examples=convert_text_to_examples(textCSV) <3>
    features=convert_examples_to_features(tokenizer,examples)

    pred=functionBert(features) <4>

    tags=TextBlob(examples[0].text).tags <5>

    vectors=[]
    text=""
    for i in range(nb_words): <6>
        if tags[i][1] in ['NNP','NN','NNS']:
            vectors.append(pred[0][0][i])
            text+=words[i]+" "

    # Words: money bank bank account river bank

    same_bank  = 1 - cosine(vectors[1], vectors[2]) <7>
    other_bank = 1 - cosine(vectors[1], vectors[5]) <8>

    print('Vector similarity for  *similar*  meanings:
[CA]    %.2f' % same_bank) <9>
    print('Vector similarity for *different* meanings:  %.2f' % other_bank)

    return (vectors, text) <10>
