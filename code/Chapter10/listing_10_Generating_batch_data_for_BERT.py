from keras_bert gen_batch_inputs

def BertGenerator(paired_sentences, tokenD, tokenL): <1>
    while True: <2>
        yield gen_batch_inputs( <3>
            paired_sentences,
            tokenD,
            tokenL,
            seq_len=200,
            mask_rate=0.3,
            swap_sentence_rate=0.5,
        )
