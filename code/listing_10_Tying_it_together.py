def main(textCSV): <1>
    bert_path="https://tfhub.dev/google/
[CA] bert_uncased_L-12_H-768_A-12/1" <2>

    tokenizer = create_tokenizer_from_hub_module(bert_path) <3>

    functionBert=createBertFunction() <4>
    (vectors,text)=readOutBertModel(tokenizer,
[CA] functionBert, textCSV) <5>

    plotTSNE(vectors,text) <6>
