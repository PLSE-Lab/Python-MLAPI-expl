#!/usr/bin/env python
# coding: utf-8

# # References 
# * [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
# * [Matplotlib Gallery](https://matplotlib.org/gallery/index.html)

# # Content
# * Compress the 4 features of Iris Dataset to 2 features using Autoencoder
# * Visualize training using TensorBoard
# * Plot the obtained 2 features and assign different colors to different species
# 
# # Issues 
# * Reconstructed data is not similar to input data.
# 
# # To be Done
# * Try to use dropout in encoding layers
#     * Dropout: A Simple Way to Prevent Neural Networks from Overfitting - http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf
# * Try different activation functions
# 
# # What's New
# * Visualized weights
#     * Initializing weights and biases in between 0 and 1 is giving better results than default initialization (which is glorot_uniform initialization)
# * Set a threshold on error
# * Tried to reduce batch_size in sgd
#     * Result is improved drastically
# * Added one more layer of length 8 on both sides of middle layer
#     * Model stopped training due to some unknown reasons

# In[ ]:


# import shutil
# shutil.rmtree('/tmp/autoencoder')


# In[ ]:


import keras
class Histories(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.weights = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        nx = []
        x = [x.reshape(-1) for x in self.model.get_weights()]
        for xi in x :
            nx += list(xi)
        self.weights.append(nx)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Importing Data

# In[ ]:


data = pd.read_csv("../input/Iris.csv")
x_train, x_test, y_train, y_test = train_test_split(data[['SepalLengthCm', 'SepalWidthCm',
                                                          'PetalLengthCm', 'PetalWidthCm']],
                                                    data['Species'],test_size=0.1, random_state=1)


# In[ ]:


x_train.head()


# ## Launching TensorBoard

# In[ ]:


# !wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
# !unzip ngrok-stable-linux-amd64.zip
# LOG_DIR = '/tmp/autoencoder' # Here you have to put your log directory
# get_ipython().system_raw(
#     'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
#     .format(LOG_DIR)
# )
# get_ipython().system_raw('./ngrok http 6006 &')
# ! curl -s http://localhost:4040/api/tunnels | python3 -c \
#     "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"


# ## Encoder and Decoder

# In[ ]:


from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard

# this is the size of our encoded representations
encoding_dim = 2
input_dim = 4

# this is our input placeholder
input_img = Input(shape=(input_dim,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(input_dim, activation='relu')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (2-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(loss='mean_squared_error', optimizer='sgd')


from keras.utils.vis_utils import plot_model
plot_model(autoencoder, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

from matplotlib import pyplot as plt
import matplotlib.image as mpimg
plt.axis("off")
plt.imshow(mpimg.imread('model_plot.png'))
plt.show()


# In[ ]:


x_hist = Histories()
while (True):
    x_hist = Histories()
    # this is our input placeholder
    input_img = Input(shape=(input_dim,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(input_dim, activation='relu')(encoded)
    
    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)

    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)

    # create a placeholder for an encoded (2-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(loss='mean_squared_error', optimizer='sgd')
    
    weight_list = []
    
    history = autoencoder.fit(x_train, x_train,
                    epochs=50,
                    batch_size=135,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    verbose=0,
                   callbacks=[TensorBoard(log_dir='/tmp/autoencoder'),x_hist])
    
    if (autoencoder.history.history['loss'][-1] < 1):
        break

# encode and decode some data points
# note that we take them from the *test* set
encoded_datapoints = encoder.predict(x_test)
decoded_datapoints = decoder.predict(encoded_datapoints)

print('Original Datapoints :')
print(x_test)
print('Reconstructed Datapoints :')
print(decoded_datapoints)


# ## Plotting Encoded Features

# In[ ]:


encoded_dataset = encoder.predict(data[['SepalLengthCm', 'SepalWidthCm','PetalLengthCm', 'PetalWidthCm']])

plt.scatter(encoded_dataset[:,0], encoded_dataset[:,1], c=data['Species'].astype('category').cat.codes)
plt.show()


# In[ ]:


autoencoder.get_weights()


# In[ ]:


from matplotlib import pyplot as plt
import matplotlib.image as mpimg
plt.axis("off")
plt.imshow(mpimg.imread('model_plot.png'))
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# ### Visualize weights

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

for i in range(len(x_hist.weights[0])):
    plt.plot(list(range(len(x_hist.weights))), [x[i] for x in x_hist.weights])

plt.show()


# ## For single iteration visualizing weights

# In[ ]:


x_hist = Histories()
# this is our input placeholder
input_img = Input(shape=(input_dim,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(input_dim, activation='relu')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (2-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(loss='mean_squared_error', optimizer='sgd')

weight_list = []

history = autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=135,
                shuffle=True,
                validation_data=(x_test, x_test),
                verbose=0,
               callbacks=[TensorBoard(log_dir='/tmp/autoencoder'),x_hist])


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

for i in range(len(x_hist.weights[0])):
    plt.plot(list(range(len(x_hist.weights))), [x[i] for x in x_hist.weights])

plt.show()


# In[ ]:


# encode and decode some data points
# note that we take them from the *test* set
encoded_datapoints = encoder.predict(x_test)
decoded_datapoints = decoder.predict(encoded_datapoints)

print('Original Datapoints :')
print(x_test)
print('Reconstructed Datapoints :')
print(decoded_datapoints)


# ###  Effect of initializing weights and biases in between 0 and 1

# In[ ]:


x_hist = Histories()
# this is our input placeholder
input_img = Input(shape=(input_dim,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu',
               kernel_initializer=keras.initializers.RandomUniform(minval=0, maxval=1, seed=None),
                bias_initializer=keras.initializers.RandomUniform(minval=0, maxval=1, seed=None))(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(input_dim, activation='relu',
               kernel_initializer=keras.initializers.RandomUniform(minval=0, maxval=1, seed=None),
                bias_initializer=keras.initializers.RandomUniform(minval=0, maxval=1, seed=None))(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (2-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(loss='mean_squared_error', optimizer='sgd')

weight_list = []

history = autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=135,
                shuffle=True,
                validation_data=(x_test, x_test),
                verbose=0,
               callbacks=[TensorBoard(log_dir='/tmp/autoencoder'),x_hist])

import matplotlib.pyplot as plt
import numpy as np

for i in range(len(x_hist.weights[0])):
    plt.plot(list(range(len(x_hist.weights))), [x[i] for x in x_hist.weights])

plt.show()


# encode and decode some data points
# note that we take them from the *test* set
encoded_datapoints = encoder.predict(x_test)
decoded_datapoints = decoder.predict(encoded_datapoints)

print('Original Datapoints :')
print(x_test)
print('Reconstructed Datapoints :')
print(decoded_datapoints)
print('Training Loss : ',history.history['loss'][-1])
print('Validation Loss : ',history.history['val_loss'][-1])


# ### Effect of reducing batch size from 135 to 15

# In[ ]:


x_hist = Histories()
# this is our input placeholder
input_img = Input(shape=(input_dim,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu',
               kernel_initializer=keras.initializers.RandomUniform(minval=0, maxval=1, seed=None),
                bias_initializer=keras.initializers.RandomUniform(minval=0, maxval=1, seed=None))(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(input_dim, activation='relu',
               kernel_initializer=keras.initializers.RandomUniform(minval=0, maxval=1, seed=None),
                bias_initializer=keras.initializers.RandomUniform(minval=0, maxval=1, seed=None))(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (2-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(loss='mean_squared_error', optimizer='sgd')

weight_list = []

history = autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=15,
                shuffle=True,
                validation_data=(x_test, x_test),
                verbose=0,
               callbacks=[TensorBoard(log_dir='/tmp/autoencoder'),x_hist])

import matplotlib.pyplot as plt
import numpy as np

for i in range(len(x_hist.weights[0])):
    plt.plot(list(range(len(x_hist.weights))), [x[i] for x in x_hist.weights])

plt.show()

# Ploting encodings
encoded_dataset = encoder.predict(data[['SepalLengthCm', 'SepalWidthCm','PetalLengthCm', 'PetalWidthCm']])

plt.scatter(encoded_dataset[:,0], encoded_dataset[:,1], c=data['Species'].astype('category').cat.codes)
plt.show()


# encode and decode some data points
# note that we take them from the *test* set
encoded_datapoints = encoder.predict(x_test)
decoded_datapoints = decoder.predict(encoded_datapoints)

print('Original Datapoints :')
print(x_test)
print('Reconstructed Datapoints :')
print(decoded_datapoints)
print('Training Loss : ',history.history['loss'][-1])
print('Validation Loss : ',history.history['val_loss'][-1])


# ### Add one more layer of size 8 on both sides of middle layer

# In[ ]:


x_hist = Histories()
# this is our input placeholder
input_img = Input(shape=(input_dim,))
# "encoded" is the encoded representation of the input
encoded = Dense(8, activation='relu',
               kernel_initializer=keras.initializers.RandomUniform(minval=0, maxval=1, seed=None),
                bias_initializer=keras.initializers.RandomUniform(minval=0, maxval=1, seed=None))(input_img)

encoded = Dense(encoding_dim, activation='relu',
               kernel_initializer=keras.initializers.RandomUniform(minval=0, maxval=1, seed=None),
                bias_initializer=keras.initializers.RandomUniform(minval=0, maxval=1, seed=None))(encoded)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(8, activation='relu',
               kernel_initializer=keras.initializers.RandomUniform(minval=0, maxval=1, seed=None),
                bias_initializer=keras.initializers.RandomUniform(minval=0, maxval=1, seed=None))(encoded)

decoded = Dense(input_dim, activation='relu',
               kernel_initializer=keras.initializers.RandomUniform(minval=0, maxval=1, seed=None),
                bias_initializer=keras.initializers.RandomUniform(minval=0, maxval=1, seed=None))(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (2-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer1 = autoencoder.layers[-2]
decoder_layer2 = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer2(decoder_layer1(encoded_input)))

autoencoder.compile(loss='mean_squared_error', optimizer='sgd')

from keras.utils.vis_utils import plot_model
plot_model(autoencoder, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

weight_list = []

history = autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=135,
                shuffle=True,
                validation_data=(x_test, x_test),
                verbose=0,
               callbacks=[TensorBoard(log_dir='/tmp/autoencoder'),x_hist])

import matplotlib.pyplot as plt
import numpy as np

for i in range(len(x_hist.weights[0])):
    plt.plot(list(range(len(x_hist.weights))), [x[i] for x in x_hist.weights])

plt.show()

# Ploting encodings
encoded_dataset = encoder.predict(data[['SepalLengthCm', 'SepalWidthCm','PetalLengthCm', 'PetalWidthCm']])

plt.scatter(encoded_dataset[:,0], encoded_dataset[:,1], c=data['Species'].astype('category').cat.codes)
plt.show()


# encode and decode some data points
# note that we take them from the *test* set
encoded_datapoints = encoder.predict(x_test)
decoded_datapoints = decoder.predict(encoded_datapoints)

print('Original Datapoints :')
print(x_test)
print('Reconstructed Datapoints :')
print(decoded_datapoints)
print('Training Loss : ',history.history['loss'][-1])
print('Validation Loss : ',history.history['val_loss'][-1])

from PIL import Image
Image.open('model_plot.png')

