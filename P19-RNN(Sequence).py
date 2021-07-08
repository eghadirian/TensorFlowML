import tensorflow.keras as keras
import matplotlib.pyplot as plt 
import numpy as np

def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10)) # wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5) # + noise
    return series[..., np.newaxis].astype(np.float32)

def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])

n_steps = 50
series = generate_time_series(10000, n_steps + 10)
X_train, Y_train = series[:7000, :n_steps], series[:7000, -10:, 0]
X_valid, Y_valid = series[7000:9000, :n_steps], series[7000:9000, -10:, 0]
X_test, Y_test = series[9000:, :n_steps], series[9000:, -10:, 0]

optimizer = keras.optimizers.Adam(lr=0.01)

model1 = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20, return_sequences=True),
    keras.layers.SimpleRNN(10)
])
model1.compile(loss="mse", optimizer=optimizer, metrics=[last_time_step_mse])
history = model1.fit(X_train, Y_train, epochs=20,
    validation_data=(X_valid, Y_valid))
acc1 = model1.evaluate(X_test, Y_test, verbose=0)

model2 = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(10)
])
model2.compile(loss="mse", optimizer=optimizer, metrics=[last_time_step_mse])
history = model2.fit(X_train, Y_train, epochs=20,
    validation_data=(X_valid, Y_valid))
acc2 = model2.evaluate(X_test, Y_test, verbose=0)

Y = np.empty((10000, n_steps, 10)) # each target is a sequence of 10D vectors
for step_ahead in range(1, 10 + 1):
  Y[:, :, step_ahead - 1] = series[:, step_ahead:step_ahead + n_steps, 0]
Y_train = Y[:7000] #data for TimeDistributed
Y_valid = Y[7000:9000]
Y_test = Y[9000:]

model3 = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(10))
])
model3.compile(loss="mse", optimizer=optimizer, metrics=[last_time_step_mse])
history = model3.fit(X_train, Y_train, epochs=20,
    validation_data=(X_valid, Y_valid))
acc3 = model3.evaluate(X_test, Y_test, verbose=0)

model4 = keras.models.Sequential([
    keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.LSTM(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(10))
])
model4.compile(loss="mse", optimizer=optimizer, metrics=[last_time_step_mse])
history = model4.fit(X_train, Y_train, epochs=20,
    validation_data=(X_valid, Y_valid))
acc4 = model4.evaluate(X_test, Y_test, verbose=0)

model5 = keras.models.Sequential([
    keras.layers.Conv1D(filters=20, kernel_size=4, strides=2, padding="valid",
        input_shape=[None, 1]),
    keras.layers.GRU(20, return_sequences=True),
    keras.layers.GRU(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(10))
])
model5.compile(loss="mse", optimizer=optimizer, metrics=[last_time_step_mse])
history = model5.fit(X_train, Y_train[:, 3::2], epochs=20,
    validation_data=(X_valid, Y_valid[:, 3::2])) #kernelsize-1(valid padding)::stridesS
acc5 = model5.evaluate(X_test, Y_test[:, 3::2], verbose=0)

model6 = keras.models.Sequential()
model6.add(keras.layers.InputLayer(input_shape=[None, 1]))
for rate in (1, 2, 4, 8) * 2:
    model6.add(keras.layers.Conv1D(filters=20, kernel_size=2, padding="causal",
        activation="relu", dilation_rate=rate))
model6.add(keras.layers.Conv1D(filters=10, kernel_size=1))
model6.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
history = model6.fit(X_train, Y_train, epochs=20,
    validation_data=(X_valid, Y_valid))
acc6 = model6.evaluate(X_test, Y_test, verbose=0)

print('MeanSquareErrors:\nModel1: {}\nModel2: {}\nModel3: {}\nModel4: {}\nModel5: {}\nModel6: {}'\
    .format(acc1[1], acc2[1], acc3[1], acc4[1], acc5[1], acc6[1]))