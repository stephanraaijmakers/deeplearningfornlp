def main(trainCSV, testCSV, valCSV): <1>

 bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1" <2>

 tokenizer = create_tokenizer_from_hub_module(bert_path) <3>

 [trainData,testData,validationData]=loadData(trainCSV, testCSV, valCSV,
  [CA]tokenizer) <4>

 finetuneBertModel(trainData) <5>
