from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)
X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
input_A = keras.layers.Input(shape = [5])
input_B = keras.layers.Input(shape = [6])
hidden1 = keras.layers.Dense(30, activation='relu', kernel_regularizer=keras.regulizers.l2(0.01))(input_B)
leaky_relu = keras.layers.LeakyReLU(alpha=0.1)
def elu(z):
    return z or 1*(np.exp(z)-1)
hidden2 = keras.layers.Dense(30, activation=leaky_relu, kernel_initilizer='he_normal')(hidden1)
# by default it uses Glorot initializer
concat = keras.layers.Concatenate()([input_A, hidden2])
output = keras.layers.Dense(1)(concat)
aux_output = keras.layers.Dense(1)(hidden2) # regularization
model = keras.models.Model(inputs=[input_A, input_B], outputs=[output, aux_output])
model.save('my_model.h5')
model = keras.models.load_model('my_model.h5')
checkpoint = keras.callbacks.ModelCheckpoint('my_model.h5',
                                             save_best_only=True)
# saves model during training at regular intervals, def = end of each epoch
# only saves the best validated model
early_stopping = keras.callbacks.EarlyStopping(patience=10,
                                               restore_best_weights=True)
# it will stop training if there is nor progress after 'patience' number of epochs
# optionally rolls back the best weights
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print('\n val/train={}'.format(logs['val_loss']/logs['loss']))
model.compile(loss=['mse', 'mse'], loss_weights=[0.9, 0.1}, optimizer='sgd')
history = model.fit((X_train_A, X_train_B),
                    (y_train, y_train),
                    epochs = 20,
                    validation_data = ((X_valid_A, X_valid_B), (y_valid, y_valid)),
                    callbacks=[checkpoint, early_stopping, CustomCallback]) # https://keras.io/callbacks/
total_loss, main_loss, aux_loss = model.evaluate((X_test_A, X_test_B), (y_test, y_test))

