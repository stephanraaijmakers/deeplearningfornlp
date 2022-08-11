class BertLayer(tf.keras.layers.Layer):

    def __init__( <1>
        self,
        n_fine_tune_layers=12,<2>
        bert_path=
        [CA]"https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",<3>
        **kwargs
    ):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True <4>
        self.output_size = 768 <5>
        self.bert_path = bert_path

        super(BertLayer, self).__init__(**kwargs)


    def build(self, input_shape): <6>
        self.bert = hub.Module( <7>
            self.bert_path,
            trainable=self.trainable,
            name=f"{self.name}_module"
        )
        trainable_vars = self.bert.variables
        trainable_vars =
        [CA][var for var in trainable_vars if not "/cls/" in var.name] <7>
        trainable_vars = trainable_vars[-self.n_fine_tune_layers :] <8>

        for var in trainable_vars: <9>
            self._trainable_weights.append(var)

        for var in self.bert.variables: <10>
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs): <11>
        inputs = [K.cast(x, dtype="int32") for x in inputs] <12>

        input_ids, input_mask, segment_ids = inputs <13>

        bert_inputs = dict( <14>
            input_ids=input_ids, input_mask=input_mask,
            [CA]segment_ids=segment_ids
        )
        result = self.bert(inputs=bert_inputs, signature="tokens",
        [CA]as_dict=True)[ <15>
            "sequence_output"
        ]
        return result

    def compute_output_shape(self, input_shape): <16>
        return (input_shape[0], self.output_size)
