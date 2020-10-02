#!/usr/bin/env python
# coding: utf-8

# Greedy layer-wise pretraining in Keras
# ==========
# 
# Apparently this method has fallen out of favor, but it's a good Keras exercise. Also, the 20 minute time limit is pretty restrictive for the number of things that have to happen here, but it appears to be working.
# 
# The code is set up for three layers of autoencoders, followed by a classification task, but the third layer is commented out due to time. Thus **this model uses two layers of pretrained autoencoders, followed by a dense layer attached to a softmax**. Dropout is used for regularization and Batch Normalization is used to speed up training. If this published kernel works the way it did in interactive mode, the final accuracy should be in the mid to high 90s.
# 
# Even if pretraining autoencoders is no longer considered a good idea, one useful trick shown in this script is how to take layers with trained weights, and "copy" them into different models using **get_weights()** and **set_weights()**.
# 
# For more on autoencoders in Keras, see The Keras Blog's [Building Autoencoders in Keras][1]
# 
#   [1]: https://blog.keras.io/building-autoencoders-in-keras.html

# In[ ]:


get_ipython().run_line_magic('env', 'KERAS_BACKEND=theano')
get_ipython().run_line_magic('reset', '')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
import keras
import keras.backend as K
from keras.layers import Input, Convolution2D, Activation, MaxPooling2D,      Dense, BatchNormalization, Dropout
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.models import Model
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler
from keras.layers.normalization import BatchNormalization

print(keras.__version__)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# ## Scaling
# 
# Scaling inputs to be between 0 and 1. That makes the decoding model simple, because we can pretend like we're working with a binary output.
# 
# 
# From the Theano docs: [binary_crossentropy][1]
# 
# ```
# crossentropy(t,o) = -(t * log(o) + (1 - t) * log(1 - o)).
# ```
# 
#   [1]: http://deeplearning.net/software/theano/library/tensor/nnet/nnet.html#theano.tensor.nnet.nnet.binary_crossentropy

# In[ ]:


N_train = 30000 # Out of 42000, to reduce processing time
train = np.genfromtxt('../input/train.csv', delimiter = ',', skip_header = 1)
training_inputs = train[0:N_train, 1:] / 255.0
training_targets = np_utils.to_categorical(train[:, int(0)])[0:N_train]

val_inputs = train[(N_train+1):42000, 1:] / 255.0
val_targets = np_utils.to_categorical(train[:, int(0)])[(N_train+1):42000]

#test = np.genfromtxt('../input/test.csv', delimiter = ',', skip_header = 1)
#test_inputs = test[:, ] / 255.0


# In[ ]:


# For 2D data (e.g. image), ordering type "tf" assumes (rows, cols, channels)
#  type "th" assumes (channels, rows, cols). See https://keras.io/backend/
print('We are using image ordering type', K.image_dim_ordering())

training_inputs = training_inputs.reshape(training_inputs.shape[0], 784)
#test_inputs = test_inputs.reshape(test_inputs.shape[0], 784)
print(training_inputs.shape)
print(val_inputs.shape)


# In[ ]:


# Layer by layer pretraining Models

# Layer 1
input_img = Input(shape = (784, ))
distorted_input1 = Dropout(.1)(input_img)
encoded1 = Dense(800, activation = 'sigmoid')(distorted_input1)
encoded1_bn = BatchNormalization()(encoded1)
decoded1 = Dense(784, activation = 'sigmoid')(encoded1_bn)

autoencoder1 = Model(input = input_img, output = decoded1)
encoder1 = Model(input = input_img, output = encoded1_bn)

# Layer 2
encoded1_input = Input(shape = (800,))
distorted_input2 = Dropout(.2)(encoded1_input)
encoded2 = Dense(400, activation = 'sigmoid')(distorted_input2)
encoded2_bn = BatchNormalization()(encoded2)
decoded2 = Dense(800, activation = 'sigmoid')(encoded2_bn)

autoencoder2 = Model(input = encoded1_input, output = decoded2)
encoder2 = Model(input = encoded1_input, output = encoded2_bn)

# Layer 3 - which we won't end up fitting in the interest of time
encoded2_input = Input(shape = (400,))
distorted_input3 = Dropout(.3)(encoded2_input)
encoded3 = Dense(200, activation = 'sigmoid')(distorted_input3)
encoded3_bn = BatchNormalization()(encoded3)
decoded3 = Dense(400, activation = 'sigmoid')(encoded3_bn)

autoencoder3 = Model(input = encoded2_input, output = decoded3)
encoder3 = Model(input = encoded2_input, output = encoded3_bn)

# Deep Autoencoder
encoded1_da = Dense(800, activation = 'sigmoid')(input_img)
encoded1_da_bn = BatchNormalization()(encoded1_da)
encoded2_da = Dense(400, activation = 'sigmoid')(encoded1_da_bn)
encoded2_da_bn = BatchNormalization()(encoded2_da)
encoded3_da = Dense(200, activation = 'sigmoid')(encoded2_da_bn)
encoded3_da_bn = BatchNormalization()(encoded3_da)
decoded3_da = Dense(400, activation = 'sigmoid')(encoded3_da_bn)
decoded2_da = Dense(800, activation = 'sigmoid')(decoded3_da)
decoded1_da = Dense(784, activation = 'sigmoid')(decoded2_da)

deep_autoencoder = Model(input = input_img, output = decoded1_da)

# Not as Deep Autoencoder
nad_encoded1_da = Dense(800, activation = 'sigmoid')(input_img)
nad_encoded1_da_bn = BatchNormalization()(nad_encoded1_da)
nad_encoded2_da = Dense(400, activation = 'sigmoid')(nad_encoded1_da_bn)
nad_encoded2_da_bn = BatchNormalization()(nad_encoded2_da)
nad_decoded2_da = Dense(800, activation = 'sigmoid')(nad_encoded2_da_bn)
nad_decoded1_da = Dense(784, activation = 'sigmoid')(nad_decoded2_da)

nad_deep_autoencoder = Model(input = input_img, output = nad_decoded1_da)


# In[ ]:


sgd1 = SGD(lr = 5, decay = 0.5, momentum = .85, nesterov = True)
sgd2 = SGD(lr = 5, decay = 0.5, momentum = .85, nesterov = True)
sgd3 = SGD(lr = 5, decay = 0.5, momentum = .85, nesterov = True)

autoencoder1.compile(loss='binary_crossentropy', optimizer = sgd1)
autoencoder2.compile(loss='binary_crossentropy', optimizer = sgd2)
autoencoder3.compile(loss='binary_crossentropy', optimizer = sgd3)

encoder1.compile(loss='binary_crossentropy', optimizer = sgd1)
encoder2.compile(loss='binary_crossentropy', optimizer = sgd1)
encoder3.compile(loss='binary_crossentropy', optimizer = sgd1)

deep_autoencoder.compile(loss='binary_crossentropy', optimizer = sgd1)
nad_deep_autoencoder.compile(loss='binary_crossentropy', optimizer = sgd1)

# What will happen to the learnning rates under this decay schedule?
lr = 5
for i in range(12):
    lr = lr - lr * .15
    print(lr)    


# In[ ]:


autoencoder1.fit(training_inputs, training_inputs,
                nb_epoch = 8, batch_size = 512,
                validation_split = 0.30,
                shuffle = True)


# In[ ]:


first_layer_code = encoder1.predict(training_inputs)
print(first_layer_code.shape)


# In[ ]:


autoencoder2.fit(first_layer_code, first_layer_code,
                nb_epoch = 8, batch_size = 512,
                validation_split = 0.25,
                shuffle = True)


# In[ ]:


#second_layer_code = encoder2.predict(first_layer_code)
#print(second_layer_code.shape)


# In[ ]:


# Not enough time!!
#autoencoder3.fit(second_layer_code, second_layer_code,
#                nb_epoch = 8, batch_size = 512,
#                validation_split = 0.30,
#                shuffle = True)


# In[ ]:


# Setting the weights of the deep autoencoder
#deep_autoencoder.layers[1].set_weights(autoencoder1.layers[2].get_weights()) # first dense layer
#deep_autoencoder.layers[2].set_weights(autoencoder1.layers[3].get_weights()) # first bn layer
#deep_autoencoder.layers[3].set_weights(autoencoder2.layers[2].get_weights()) # second dense layer
#deep_autoencoder.layers[4].set_weights(autoencoder2.layers[3].get_weights()) # second bn layer
#deep_autoencoder.layers[5].set_weights(autoencoder3.layers[2].get_weights()) # thrird dense layer
#deep_autoencoder.layers[6].set_weights(autoencoder3.layers[3].get_weights()) # third bn layer
#deep_autoencoder.layers[7].set_weights(autoencoder3.layers[4].get_weights()) # first decoder
#deep_autoencoder.layers[8].set_weights(autoencoder2.layers[4].get_weights()) # second decoder
#deep_autoencoder.layers[9].set_weights(autoencoder1.layers[4].get_weights()) # third decoder

# Setting up the weights of the not-as-deep autoencoder
nad_deep_autoencoder.layers[1].set_weights(autoencoder1.layers[2].get_weights()) # first dense layer
nad_deep_autoencoder.layers[2].set_weights(autoencoder1.layers[3].get_weights()) # first bn layer
nad_deep_autoencoder.layers[3].set_weights(autoencoder2.layers[2].get_weights()) # second dense layer
nad_deep_autoencoder.layers[4].set_weights(autoencoder2.layers[3].get_weights()) # second bn layer
nad_deep_autoencoder.layers[5].set_weights(autoencoder2.layers[4].get_weights()) # second decoder
nad_deep_autoencoder.layers[6].set_weights(autoencoder1.layers[4].get_weights()) # third decoder


# In[ ]:


# you can see the degredation by uncommenting these one at a time and plotting
#decoded_inputs = autoencoder1.predict(training_inputs[0:25, ])
decoded_inputs = nad_deep_autoencoder.predict(training_inputs[0:25,])
#decoded_inputs = deep_autoencoder.predict(training_inputs[0:25,])
decoded_inputs.shape

fig = plt.figure(figsize = (8, 8))
fig.suptitle('Deep autoencoder reconstructions', fontsize=24, fontweight='bold')

ax1 = fig.add_subplot(231)
plt.imshow(training_inputs[2].reshape(28, 28))

ax2 = fig.add_subplot(234)
plt.imshow(decoded_inputs[2].reshape(28, 28))

ax3 = fig.add_subplot(232)
plt.imshow(training_inputs[6].reshape(28, 28))

ax4 = fig.add_subplot(235)
plt.imshow(decoded_inputs[6].reshape(28, 28))

ax5 = fig.add_subplot(233)
plt.imshow(training_inputs[4].reshape(28, 28))

ax6 = fig.add_subplot(236)
plt.imshow(decoded_inputs[4].reshape(28, 28))


# ## On to "fine tuning" for classification
# Although it will have to be more than fine tuning

# In[ ]:


dense1 = Dense(500, activation = 'relu')(nad_decoded1_da)
dense1_drop = Dropout(.3)(dense1)
#dense1_bn = BatchNormalization()(dense1_drop)
dense2 = Dense(10, activation = 'sigmoid')(dense1_drop)

classifier = Model(input = input_img, output = dense2)
sgd4 = SGD(lr = .1, decay = 0.001, momentum = .95, nesterov = True)
classifier.compile(loss='categorical_crossentropy', optimizer = sgd4, metrics=['accuracy'])
   
classifier.fit(training_inputs, training_targets,
                nb_epoch = 6, batch_size = 600,
                validation_split = 0.25,
                shuffle = True)


# In[ ]:


val_preds = classifier.predict(val_inputs)
predictions = np.argmax(val_preds, axis = 1)
true_digits = np.argmax(val_targets, axis = 1)
predictions[0:25]


# In[ ]:


n_correct = np.sum(np.equal(predictions, true_digits).astype(int))
total = float(len(predictions))
print("Validation Accuracy:", round(n_correct / total, 3))

