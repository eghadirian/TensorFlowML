import tensorflow as tf
from tensorflow import keras
# custom loss function
# defining your own loss function: Huber
def create_huber(threshold=1.):
    def huber(y_true, y_pred):
    # use vectorized operation and tf functions
        error = y_true - y_pred
        is_small_error = tf.abs(error)<threshold
        squared_loss = tf.square(error)/2
        linear_loss = threshold*tf.abs(error) - threshold**2/2.
        return tf.where(is_small_error, squared_loss, linear_loss)
    return huber
model.compile(loss=create_huber(0.3), optimizer='nadam')
# loading the model with custom function
model = keras.models.load_model('my_model.h5', custom_objects={'huber':create_huber(0.3)})
# if you habe configured class, you need to save the threshold or you can do this:
class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold, **kwargs):
        self.threshold = threshold
        super.__init__(**kwargs)
    def call(self, y_true, y_pred): # call for models, layers, activations, losses
        error = y_true - y_pred
        is_small_error = error<self.threshold
        squared_loss = tf.square(error)/2.
        linear_loss = self.threshold*tf.abs(error) - self.threshold**2/2.
        return tf.where(is_small_error, squared_loss, linear_loss)
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'threshold':self.threshold}
model.compile(loss=HuberLoss(0.3), optimizer='nadam')
model.save('my_model.h5')
model = keras.models.load_model('my_model.h5', custom_objects={'HuberLoss':HuberLoss})
# custom activation function
def softplus(z):
    return tf.math.log(tf.exp(z)+1)
# custom initializer
def glorot_initializer(shape, dtype=tf.float32):
    stddev = tf.sqrt(2./(shape[0]+shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)
# regularizer
def l1_regularizer(weights):
    return tf.reduce_sum(tf.abs(0.01*weights))
# custom constraint
def positive_weights(weights):
    return tf.where(weights<0., tf.zeros_like(weights), weights)

layer = keras.layers.Dense(40,
                           activation = softplus,
                           jernel_initializer = glorot_initializer,
                           kernel_regularizer = l1_regularizer,
                           kernel_constraint = positive_weights)
# if there are hyperparameter to be saved we should do subclassing:
class MyRegularizer(keras.regularizers.Regularizer):
    def __init__(self, factor):
        self.factor = factor
    def __call__(self, weights): # __call__ for regularizers, constarints, initializers
        return tf.reduce_sum(tf.abs(self.factr*weights))
    def get_config(self):
        return {'factor':self.factor}
# custom metrics
class HuberMetrics(keras.metrics.Metric):
    def __init__(self, threshold=1., **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.huber = create_huber(threshold)
        self.total = self.add_weights('total', initializer='zeros')
        self.count = self.add_weigts('count', initializer='zeros')
    def update_state(self, y_true, y_pred, sample_weight=None):
        metric = self.huber(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign(tf.cast(tf.size(y_true), tf.float32))
    def result(self):
        return self.total/self.count
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'threshold':self.threshold}
    