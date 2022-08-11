class BertLayer(tf.keras.layers.Layer):

    def __init__( 
        self,
        n_fine_tune_layers=12,
        bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
        **kwargs
    ):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True 
        self.output_size = 768 
        self.bert_path = bert_path

        super(BertLayer, self).__init__(**kwargs)


    def build(self, input_shape): 
        self.bert = hub.Module( 
            self.bert_path,
            trainable=self.trainable,
            name=f"{self.name}_module"
        )
        trainable_vars = self.bert.variables
        trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name] 
        trainable_vars = trainable_vars[-self.n_fine_tune_layers :] 

        for var in trainable_vars: 
            self._trainable_weights.append(var)

        for var in self.bert.variables: 
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs): 
        inputs = [K.cast(x, dtype="int32") for x in inputs] 
        input_ids, input_mask, segment_ids = inputs 
        bert_inputs = dict( 
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[ 
            "sequence_output"
        ]
        return result

    def compute_output_shape(self, input_shape): 
        return (input_shape[0], self.output_size)

