import tensorflow as tf
from tensorflow import keras
import numpy as np

mnist = keras.datasets.fashion_mnist
(training_image, training_label), (test_image, test_label) = mnist.load_data()
training_image=training_image.reshape(60000, 28, 28, 1)
test_image=test_image.reshape(60000, 28, 28, 1)
training_image, test_image = training_image/255., test_image/255.
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.6):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True
callbacks = myCallback()
model = keras.Sequential([
    keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_image, training_label, epochs=5, callbacks=[callbacks])
test_loss, test_acc = model.evaluate(test_image, test_label)


