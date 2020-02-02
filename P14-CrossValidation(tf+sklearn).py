from tensorflow import keras
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[0]):
    model= keras.models.Sequential()
    options = {'input_shape': input_shape}
    for layers in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation='relu', **options))
        options = {}
    model.add(keras.layers.Dense(1, **options))
    optimizer = keras.optimizers.SGD(lr=learning_rate, decay = 1e-4, clipvalue=1., momentum = 0.9, nesterov=True)
    model.compile(loss='mse', optimizer=optimizer)
    return model
def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0*0.1**(epoch/s)
    return exponential_decay_fn
exponential_decay_fn = exponential_decay(0.01, 20)
lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)
keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)
param_distribs = {
    'n_hidden':[0,1,2,3],
    'n_neurons':np.arange(1,100),
    'learnig_rate':reciprocal(3e-4, 3e-2)
}
rnd_search_CV = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)
rnd_search_CV.fit(X_train, y_train,
                  epochs=100,
                  validation_data=(X_valid, y_valid),
                  callbacks=[keras.callbacks.EarlyStoping(patience=10), lr_scheduler])

