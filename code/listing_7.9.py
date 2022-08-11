
...

    self.recurrent_attention_kernel = self.add_weight(
            shape=(input_dim, input_dim),
            name='recurrent_attention_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

...

