from keras import backend as K
from keras.layers import Layer

class MyLayer(Layer): <1>

    def __init__(self, output_dim, **kwargs): <2>
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape): <3>
        self.kernel = self.add_weight(name='kernel',
                              shape=(input_shape[1], self.output_dim),
                              initializer='uniform',
                              trainable=True)
        super(MyLayer, self).build(input_shape)

    def call(self, x): <4>
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape): <5>
        return (input_shape[0], self.output_dim)
