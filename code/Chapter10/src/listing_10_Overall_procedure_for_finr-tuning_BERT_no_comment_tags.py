def main(trainCSV, testCSV, valCSV): 
 bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1" 

 tokenizer = create_tokenizer_from_hub_module(bert_path) 

 [trainData,testData,validationData]=loadData(trainCSV, testCSV, valCSV,tokenizer) 

 finetuneBertModel(trainData) 
