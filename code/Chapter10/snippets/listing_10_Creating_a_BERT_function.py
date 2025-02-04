import tensorflow.keras.backend as K

def createBertFunction():
  max_seq_length=256
  in_id = tf.keras.layers.Input(shape=(max_seq_length,),
  [CA]name="input_ids") <1>
  in_mask = tf.keras.layers.Input(shape=(max_seq_length,),
  [CA]name="input_masks")
  in_segment = tf.keras.layers.Input(shape=(max_seq_length,),
  [CA]name="segment_ids")

  bert_input = [in_id, in_mask, in_segment] <2>

  bert_output = BertLayer(n_fine_tune_layers=10)(bert_input) <3>

  func = K.function([bert_input], [bert_output]) <4>

  session = K.get_session() <5>
  init = tf.compat.v1.global_variables_initializer()
  session.run(init)

  return func <6>
