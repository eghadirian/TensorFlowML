from tensorflow import keras

class WideAndDenseModel(keras.models.Model):
    def __init__(self, units=30, activation='relu', **kwargs):
        super().__init__(**kwargs) # initializes everthing based on super class: keras.models.Model
        self.hidden1 = keras.layers.Dense(units, activation=activation)
        self.hidden2 = keras.layers.Dense(units, activation=activation)
        self.BN = keras.layers.BatchNormalization()
        self.main_output = keras.layers.Dense(1)
        self.aux_output = keras.layers.Dense(1)
    def call(self, inputs):
        '''
        architecture is hidden in Call method
        It cannot be saved or cloned
        '''
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden1 = self.BN(hidden1)
        hidden2 = self.hidden2(hidden1)
        hidden2 = self.BN(hidden2)
        concat = keras.layers.concat([input_A, hidden2])
        concat = self.BN(concat) # not suitable for RNNs
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output

model = WideAndDenseModel()
