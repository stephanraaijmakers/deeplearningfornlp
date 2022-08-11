def build(self, input_shape):

...
    self.attention_kernel = self.add_weight(name='attention',
                                 shape=(input_dim,input_dim),
                                 initializer='uniform',
                                 trainable=True)
...
