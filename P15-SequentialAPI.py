from tensorflow import keras
from functools import partial
import numpy as np

RegularizerDense = partial(keras.layers.Dense,
                           activation='elu', # activation
                           kernel_initializer='he_normal', # initializer
                           kernel_regularizer=keras.regularizers.l2(0.01), # regularization
                           kernel_constraint=keras.constraints.max_norm(1)) # regularization
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dropout(rate=0.2),
    RegularizerDense(100),
    keras.layers.Dropout(rate=0.2),
    RegularizerDense(100),
    keras.layers.Dropout(rate=0.2),
    RegularizerDense(10, activation='softmax', kernel_initializer='glorot_uniform')
])
# boosting the drop out model without retraining it: Monte-Carlo Dropout
# keeps dropout on and samples multiple prediction
# acts after model.fit
with keras.backened.learning_phase_scope(1):
# keeps training steps all on during the code execution
# dropouts and batch normalization etc are all on
    y_probas = np.stack[model.predict(X_test_scaled) for sample in range(100)] # 100 ==> # of monte-carlo samples
y_probas = y_probas.mean(axis=0)

# if only want to enable MCDroput not N ets, replace dropout with:
class MCDropout(keras.models.Dropout):
# also acts like regularizers as well
# then we don't need "with keras.backened.learning_phase_scope(1):"
    def call(self, inputs):
        return super().call(inputs, training=True)

