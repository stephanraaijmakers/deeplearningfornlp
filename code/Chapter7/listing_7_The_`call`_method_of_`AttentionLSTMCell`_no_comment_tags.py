def call(self, inputs, states, training=None):
                ...
                   x_i = K.dot(inputs_i, self.kernel_i) 
                   x_a = K.dot(inputs_a, self.kernel_a)
                   x_f = K.dot(inputs_f, self.kernel_f)
                   x_c = K.dot(inputs_c, self.kernel_c)
                   x_o = K.dot(inputs_o, self.kernel_o)
                   ...

                   i = self.recurrent_activation(x_i + K.dot(h_tm1_i,
                        self.recurrent_kernel_i))
                   f = self.recurrent_activation(x_f + K.dot(h_tm1_f,
                        self.recurrent_kernel_f))
                   # Attention
                   a = self.recurrent_attention_activation(x_a + K.dot(
                        h_tm1_a,self.recurrent_kernel_a)) 

                   attP=a*self.attention_activation(x_a+K.dot(
                        h_tm1_a,self.recurrent_kernel_a)) 

                   c = f * c_tm1 +i * self.activation(x_c + K.dot(
                        h_tm1_c, self.recurrent_kernel_c))+attP

                   o = self.recurrent_activation(x_o + K.dot(h_tm1_o,
                        self.recurrent_kernel_o))
               ...

               if self.attention_flag: 
                   return a, [a, c]
               else:
                   return h,[h,c]
