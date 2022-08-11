import tensorflow_hub as hub
import tensorflow as tf
from bert import bert_tokenization

def create_tokenizer_from_hub_module(bert_hub_path):
  with tf.Graph().as_default(): 
    bert_module = hub.Module(bert_hub_path) 
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.compat.v1.Session() as sess: 
      vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
  return bert_tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case) 
