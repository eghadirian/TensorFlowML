import tensorflow as tf
from tensorflow import keras

# No Weight Layer: Flatten, ReLU
exponential_layer = keras.layers.Lambda(lambda x:tf.exp(x))
# with weights
class MyDense(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
    def build(self, batch_inout_shape):
        self.kernel = self.add_weight(
            name='kernel',shape=[batch_inout_shape[-1], self.units],
            initializer='glorot_normal'
        )
        self.bias = self.add_weight(
            name='bias', shape=[self.units], initializer='zeros',
        )
        super().build(batch_inout_shape)
    def call(self, X, training=None):
        if training:
            return self.activation(X @ self.kernel + self.bias)
        else:
            return X
    def compute_output_shape(self, batch_input_shape):
        return tf.TensotShape(batch_input_shape.as_list()[:-1]+[self.units])
    def get_congig(self):
        base_config = super().get_config()
        return {**base_config,
            'units':self.units,
            'activation':keras.activation.serialize(self.activation)}

class MyMultilayer(keras.layers.Layer):
    def call(self, X):
        X1, X2 = X # multi-input
        return [X1+X2, X1/X2] # multi-output
    def compute_output_shape(self, batch_input_shape):
        b1, b2 = batch_input_shape
        return [b1, b1]

