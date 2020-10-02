import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
    
x = Input(shape=(1,), name='x')

y = Input(shape=(1,), name='y')

y_pred = Dense(1, name='y_pred')(x)

model = Model(inputs=[x, y], outputs=[y_pred])

mse = MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
loss = lambda: mse(y_pred, y)

optimizer = tf.keras.optimizers.Adam()
train_op = optimizer.minimize(loss, model.trainable_variables, name="train")