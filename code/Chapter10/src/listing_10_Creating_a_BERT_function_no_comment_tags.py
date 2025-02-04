import tensorflow.keras.backend as K

def createBertFunction():
  max_seq_length=256
  in_id = tf.keras.layers.Input(shape=(max_seq_length,),name="input_ids") 
  in_mask = tf.keras.layers.Input(shape=(max_seq_length,),name="input_masks")
  in_segment = tf.keras.layers.Input(shape=(max_seq_length,),name="segment_ids")

  bert_input = [in_id, in_mask, in_segment] 

  bert_output = BertLayer(n_fine_tune_layers=10)(bert_input) 

  func = K.function([bert_input], [bert_output]) 

  session = K.get_session() 
  init = tf.compat.v1.global_variables_initializer()
  session.run(init)

  return func 
