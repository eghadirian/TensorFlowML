import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m, n = housing.data.shape
housing_plus_bias = np.c_[np.ones((m, 1)), housing.data]
X = tf.constant(housing_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name='y')
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.linalg.inv(tf.matmul(XT, X)), XT), y)