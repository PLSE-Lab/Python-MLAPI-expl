#!/usr/bin/env python
# coding: utf-8

# This kernel is a longer, statisticaly motivated, comparison between the use of sine and ReLU as activation functions in Neural Networks, based on this [notebook](https://www.kaggle.com/aakashnain/siren) by [Nain](https://www.kaggle.com/aakashnain)
# 
# The following text is a direct quote from his kernel:
# "A new [paper](https://arxiv.org/pdf/2006.09661.pdf) that was recently preseneted in CVPR 2020 proposed to use `sine` activation function as compared to `relu`. As shown here in the [video](https://www.youtube.com/watch?time_continue=3&v=Q2fLWGBeaiI&feature=emb_logo), the representations learned by the `sine` activations are remarkable as compared to any other activation. The paper also suggests that using `sine` activation, we get better convergence.
# 
# To this end, I thought to put up a few small scale experiments to check the validity of the claims. Here I am going to demonstrate the usage on `MNSIT` and `CIFAR` dataset."

# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(16, 9))


# # MNIST

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
# 1. Allow the use of different activation functions, in this case we will compare `relu` (as in the original code) to `sin` (as in the siren paper)
# 2. Change initializer from `glorot_uniform` to `he_uniform`, as to match both the initializer in the Siren paper ($U \big[-\sqrt{\frac{6}{n}}, \sqrt{\frac{6}{n}}\big]$) and the best practice of using He init for networks with ReLU activation

# In[ ]:


def create_mnist_model(act_fn=tf.nn.relu, input_shape=(28, 28, 1), num_classes=10):
    model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation=act_fn, kernel_initializer="he_uniform"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation=act_fn, kernel_initializer="he_uniform"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
    return model


# In[ ]:


sample_model = create_mnist_model()
sample_model.summary()


# ## ReLU activation

# In[ ]:


batch_size = 128
epochs = 15
losses, acc = [], []
for i in range(20):
    model = create_mnist_model()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=0)
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f"{i+1}th Test loss:", score[0])
    print(f"{i+1}th Test accuracy:", score[1])
    losses.append(score[0])
    acc.append(score[1])


# In[ ]:


print(f'Average acc: {np.mean(acc)}')
print(f'Acc Standard Deviation: {np.std(acc)}')


# ## Sine Activation

# In[ ]:


batch_size = 128
epochs = 15
sin_losses, sin_acc = [], []
for i in range(20):
    model = create_mnist_model(tf.math.sin)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=0)
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f"{i+1}th Test loss:", score[0])
    print(f"{i+1}th Test accuracy:", score[1])
    sin_losses.append(score[0])
    sin_acc.append(score[1])


# In[ ]:


print(f'Average acc: {np.mean(sin_acc)}')
print(f'Acc Standard Deviation: {np.std(sin_acc)}')


# ## Comparison

# In[ ]:


sns.distplot(sin_acc, color='b')
sns.distplot(acc, color='r')
plt.show()


# In[ ]:


sns.distplot(sin_losses, color='b')
sns.distplot(losses, color='r')
plt.show()


# **Conclusion:** 
# 
# There is no statisticaly relevant difference between the performance of the two activations 

# # CIFAR-10

# In[ ]:


# Model / data parameters
num_classes = 10
input_shape = (32, 32, 3)


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


def create_cifar_model(act_fn=tf.nn.relu, input_shape=(32, 32, 3), num_classes=10):
    model = keras.models.Sequential()
    model.add(layers.Conv2D(32,
                 (3, 3),
                 padding='same',
                 kernel_initializer="he_uniform",
                 activation=act_fn,
                 input_shape=x_train.shape[1:]))
    model.add(layers.Conv2D(32,
                 (3, 3),
                 kernel_initializer="he_uniform",
                 activation=act_fn))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64,
                 (3, 3),
                 padding='same',
                 kernel_initializer="he_uniform",
                 activation=act_fn))
    model.add(layers.Conv2D(64,
                 (3, 3),
                 kernel_initializer="he_uniform",
                 activation=act_fn))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(512, kernel_initializer="he_uniform", activation=act_fn))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))
    
    return model


# In[ ]:


sample_model = create_cifar_model()
sample_model.summary()


# ## ReLU activation

# In[ ]:


epochs=15
losses, acc = [], []

for i in range(20):
    model = create_cifar_model()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=0)
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f"{i+1}th Test loss:", score[0])
    print(f"{i+1}th Test accuracy:", score[1])
    losses.append(score[0])
    acc.append(score[1])


# In[ ]:


print(f'Average acc: {np.mean(acc)}')
print(f'Acc Standard Deviation: {np.std(acc)}')


# ## Sine activation

# In[ ]:


epochs=15
sin_losses, sin_acc = [], []

for i in range(20):
    model = create_cifar_model(tf.math.sin)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=0)
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f"{i+1}th Test loss:", score[0])
    print(f"{i+1}th Test accuracy:", score[1])
    sin_losses.append(score[0])
    sin_acc.append(score[1])


# In[ ]:


print(f'Average acc: {np.mean(sin_acc)}')
print(f'Acc Standard Deviation: {np.std(sin_acc)}')


# ## Comparison

# In[ ]:


sns.distplot(sin_losses, color='b')
sns.distplot(losses, color='r')
plt.show()


# In[ ]:


sns.distplot(sin_acc, color='b')
sns.distplot(acc, color='r')
plt.show()


# **Conclusion:** 
# 
# ReLU did better

# # Final Thoughts
# 
# 1.  No statisticaly relevant difference was found between the performance of the two functions in MNIST, and ReLU did better in CIFAR
# 2.  This does not disagree with the paper's conclusion, as the application is in a more specific problem, and they do not claim that using sine as an activation will improve performance of all Neural Networks.
# 
# 
# **Please upvote if you liked the kernel.**
