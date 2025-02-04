import pandas as pd
from sklearn.preprocessing import LabelEncoder

def loadData(trainCSV, testCSV, valCSV, tokenizer): 
    max_seq_length=256

    train = pd.read_csv(trainCSV) 
    test = pd.read_csv(testCSV)
    val = pd.read_csv(valCSV)

    label_encoder = LabelEncoder().fit(pd.concat([train['label'],
    val['label']])) 

    y_train = label_encoder.fit_transform(pd.concat([train['label'],
    val['label']]))
    y_test = label_encoder.fit_transform(pd.concat([test['label'],
    val['label']]))
    y_val = label_encoder.fit_transform(pd.concat([train['label'],
    val['label']]))

    train_examples = convert_text_to_examples(train['text'], y_train) 
    test_examples = convert_text_to_examples(test['text'], y_test)
    val_examples = convert_text_to_examples(val['text'], y_val)

    (train_input_ids, train_input_masks, train_segment_ids, train_labels) =
    convert_examples_to_features(tokenizer, train_examples,
    max_seq_length=max_seq_length) 
    (test_input_ids, test_input_masks, test_segment_ids, test_labels) =
    convert_examples_to_features(tokenizer, test_examples,
    max_seq_length=max_seq_length)
    (val_input_ids, val_input_masks, val_segment_ids, val_labels) =
    convert_examples_to_features(tokenizer, val_examples,
    max_seq_length=max_seq_length)

    return [(train_input_ids,train_input_masks,train_segment_ids,
        train_labels),
      (test_input_ids,test_input_masks,test_segment_ids, test_labels),
       (val_input_ids,val_input_masks,val_segment_ids, val_labels)] 
