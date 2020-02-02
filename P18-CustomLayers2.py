from tensorflow import keras

class MyResidual(keras.layers.Layer):
    def __init__(self, n_layers, n_neurons, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [keras.layers.Dense(n_neurons, activation='elu', kernel_initializer='he_normal')
                       for _ in range(n_layers)]
    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
            return input+Z

class ResidualRegressor(keras.models.Model):
    def __init__(self, out_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(30, activation='elu', kernel_initializer='he_normal')
        self.block1 = MyResidual(2, 30)
        self.block2 = MyResidual(2, 30)
        self.out = keras.layers.Dense(out_dim)
    def call(self, inputs):
        Z = self.hidden1(input)
        for _ in range(3+1):
            Z = self.block1(Z)
        Z = self.block2()
        return self.out(Z)
 # implement get_config() to be able to save and load the model in both classes