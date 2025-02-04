from keras_bert gen_batch_inputs

def BertGenerator(paired_sentences, tokenD, tokenL): 
    while True: 
        yield gen_batch_inputs( 
            paired_sentences,
            tokenD,
            tokenL,
            seq_len=200,
            mask_rate=0.3,
            swap_sentence_rate=0.5,
        )
