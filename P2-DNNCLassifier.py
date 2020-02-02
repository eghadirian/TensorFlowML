import tensorflow as tf
from tensorflow import keras
from scipy.io import loadmat
from six.moves import urllib

mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
mnist_path = "./mnist-original.mat"
response = urllib.request.urlopen(mnist_alternative_url)
with open(mnist_path, "wb") as f:
    content = response.read()
    f.write(content)
mnist_raw = loadmat(mnist_path)
mnist = {
    "data": mnist_raw["data"].T,
    "target": mnist_raw["label"][0],
    "COL_NAMES": ["label", "data"],
    "DESCR": "mldata.org dataset: mnist-original",
}
X, y = mnist['data'], mnist['target']
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
def leaky_relu(z, alpha=0.01, name=None):
    return tf.maximum(alpha*z, z, name)
model = keras.models.Sequential=([
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D,
    keras.layers.BatchNormalization(),
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(512, activation=leaky_relu),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, epochs=5)


