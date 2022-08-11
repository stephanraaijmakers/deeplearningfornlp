import tensorflow.keras.backend as K
import tensorflow as tf

def buildBertModel():
  max_seq_length=256
  in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids") 
  in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
  in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")

  bert_inputs = [in_id, in_mask, in_segment] 

  bert_output = BertLayer(n_fine_tune_layers=10)(bert_inputs) 

  drop = keras.layers.Dropout(0.3)(bert_output) 
  dense = keras.layers.Dense(200, activation='relu')(bert_output)
  drop = keras.layers.Dropout(0.3)(dense)
  dense = keras.layers.Dense(100, activation='relu')(dense)
  pred = keras.layers.Dense(1, activation='sigmoid')(dense)

  session = K.get_session() 
  init = tf.compat.v1.global_variables_initializer()
  session.run(init)

  model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred) 
  model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
  model.summary()
  return model
