#!/usr/bin/env python
# coding: utf-8

# A new [paper](https://arxiv.org/pdf/2006.09661.pdf) that was recently preseneted in CVPR 2020 proposed to use `sine` activation function as compared to `relu`. As shown here in the [video](https://www.youtube.com/watch?time_continue=3&v=Q2fLWGBeaiI&feature=emb_logo), the representations learned by the `sine` activations are remarkable as compared to any other activation. The paper also suggests that using `sine` activation, we get better convergence.
# 
# To this end, I thought to put up a few small scale experiments to check the validity of the claims. Here I am going to demonstrate the usage on `MNSIT` and `CIFAR` dataset.
# 
# **Note:** I am doing some more experimentation, so keep an eye on the noetbooks section for more on it. Also, blogpost coming soon as well.

# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[ ]:


seed = 1234
np.random.seed(seed)
tf.random.set_seed(seed)


# ## MNIST

# In[ ]:


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# This example has been taken from the Keras [website](https://keras.io/examples/vision/mnist_convnet/) and the following modifications were done:
# 
# 1. Change activation function from `relu` to `sin`
# 2. Change initializer from `glorot_uniform` to `he_uniform`

# In[ ]:


model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation=tf.math.sin, kernel_initializer="he_uniform"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation=tf.math.sin, kernel_initializer="he_uniform"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()


# In[ ]:


batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)


# In[ ]:


score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


# **Conclusion:** 
# 
# This network performed far better than the original one with `relu` activations. This network achieved much lower loss `(~0.25 vs ~0.26)` on the test set. The test accuracy is also much better `(~99 vs ~991xx)`

# ## CIFAR-10

# In[ ]:


# Model / data parameters
num_classes = 10
input_shape = (32, 32, 1)


# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# This example has been taken from the old keras [website](https://keras.io/examples/cifar10_cnn/) and the following modifications were done:
# 
# 1. Change activation function from `relu` to `sin`
# 2. Change initializer from `glorot_uniform` to `he_uniform`

# In[ ]:


model = keras.models.Sequential()
model.add(layers.Conv2D(32,
                 (3, 3),
                 padding='same',
                 kernel_initializer="he_uniform",
                 activation=tf.math.sin,
                 input_shape=x_train.shape[1:]))
model.add(layers.Conv2D(32,
                 (3, 3),
                 kernel_initializer="he_uniform",
                 activation=tf.math.sin))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64,
                 (3, 3),
                 padding='same',
                 kernel_initializer="he_uniform",
                 activation=tf.math.sin))
model.add(layers.Conv2D(64,
                 (3, 3),
                 kernel_initializer="he_uniform",
                 activation=tf.math.sin))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(512, kernel_initializer="he_uniform", activation=tf.math.sin))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation="softmax"))

model.summary()


# In[ ]:


epochs=25
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)


# **Conclusion:** 
# 
# Even without any `augmentation`, this network achieved the same validation accuracy `(~74%-75%)` whereas heavy augmentation is used in the original implementation. Although you can argue that with augmentation the network would take much more time to generalize as in the case of the original implementation, I would say that same holds for `overfitting`. The network isn't that bad in this case. 

# ## Final Thoughts
# 
# 1. As per the paper, it seems we are surely having faster convergence. Though I am running a few more tests and will report more results soon.
# 2. I think `sine` activation is the first **true** competitor of `relu`.
# 3. The best part is that if this holds for other experiments as well, the code changes are negligible.
# 
# 
# **Please upvote if you liked the kernel. People think I do nothing on Kagggle except bragging, your votes will help.**

# In[ ]:




