from tensorflow import keras
from keras_bert import get_model, compile_model


def buildBertModel(paired_sentences,tokenD,tokenL, model_path): 
    model = get_model( 
        token_num=len(tokenD),
        head_num=5,
        transformer_num=12,
        embed_dim=256,
        feed_forward_dim=100,
        seq_len=200,
        pos_num=200,
        dropout_rate=0.05
    )
    compile_model(model) 

    model.fit_generator( 
        generator=BertGenerator(paired_sentences,tokenD,tokenL),
        steps_per_epoch=100,
        epochs=10
    )
    model.save(model_path) 


sentences="./my-sentences.txt" 
(paired_sentences,tokenD,tokenL)=readSentencePairs(sentences)
model_path="./bert.model"
buildBertModel(paired_sentences,tokenD,tokenL,model_path)
