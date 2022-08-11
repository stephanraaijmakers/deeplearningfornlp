from tensorflow import keras
from keras_bert import get_model, compile_model

def buildBertModel(paired_sentences,tokenD,tokenL, model_path): <1>
    model = get_model( <2>
        token_num=len(tokenD),
        head_num=5,
        transformer_num=12,
        embed_dim=256,
        feed_forward_dim=100,
        seq_len=200,
        pos_num=200,
        dropout_rate=0.05
    )
    compile_model(model) <3>

    model.fit_generator( <4>
        generator=BertGenerator(paired_sentences,tokenD,tokenL),
        steps_per_epoch=100,
        epochs=10
    )
    model.save(model_path) <5>

sentences="./my-sentences.txt" <5>
(paired_sentences,tokenD,tokenL)=readSentencePairs(sentences)
model_path="./bert.model"
buildBertModel(paired_sentences,tokenD,tokenL,model_path)
