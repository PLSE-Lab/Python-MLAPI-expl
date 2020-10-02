#!/usr/bin/env python
# coding: utf-8

# ## Tiny ResNet with Keras ##
# 
# ----------
# 
# Let's solve this classic problem elegantly, using modern approaches and using as little code as possible.

# ### Imports ###
# 
# Let's import basic packages

# In[ ]:


import numpy as np                  # for working with tensors outside the network
import pandas as pd                 # for data reading and writing
import matplotlib.pyplot as plt     # for data inspection


# and all the Keras stuff we will need

# In[ ]:


from keras.layers import Input, Conv2D, Activation, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout
from keras.layers.merge import add
from keras.activations import relu, softmax
from keras.models import Model
from keras import regularizers


# and two handy sklearn functions for data preprocessing

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


# ### The model ###
# 
# The main idea of residual architecture is the shortcut connections. Instead of learning mapping
# $$\mathcal{H}(x)$$
# we learn the mapping
# $$\mathcal{F}(x) = f(x) + \mathcal{H}(x).$$
# 
# If
# $$f(x) = x$$
# then  the entire network can be written as
# $$\mathcal{F}(x) = x + \sum_{l=0}^{L-1} \mathcal{F}_{l+1} ( x_l ) $$
# which allows the error to propagate unchanged to the top of the network.
# 
# Moreover, It's easier to learn the
# $$\mathcal{F}_l(x) = 0$$
# mapping than the
# $$\mathcal{H}(x) = x$$
# mapping when it's necessary to exclude some layer from the network.

# In practice, we can't use the identity function as f everywhere because of inconsistent dimensions, so when the input and output of the H_l have the different dimensionality we will use the 2d convolution with 1x1 kernel as the f function. Let's define the basic building block which consists of two 3x3 convolutional layers with pre-activation and the shortcut connection.

# In[ ]:


def block(n_output, upscale=False):
    # n_output: number of feature maps in the block
    # upscale: should we use the 1x1 conv2d mapping for shortcut or not
    
    # keras functional api: return the function of type
    # Tensor -> Tensor
    def f(x):
        
        # H_l(x):
        # first pre-activation
        h = BatchNormalization()(x)
        h = Activation(relu)(h)
        # first convolution
        h = Conv2D(kernel_size=3, filters=n_output, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(h)
        
        # second pre-activation
        h = BatchNormalization()(x)
        h = Activation(relu)(h)
        # second convolution
        h = Conv2D(kernel_size=3, filters=n_output, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(h)
        
        # f(x):
        if upscale:
            # 1x1 conv2d
            f = Conv2D(kernel_size=1, filters=n_output, strides=1, padding='same')(x)
        else:
            # identity
            f = x
        
        # F_l(x) = f(x) + H_l(x):
        return add([f, h])
    
    return f


# And now let's define the entire model.

# In[ ]:


# input tensor is the 28x28 grayscale image
input_tensor = Input((28, 28, 1))

# first conv2d with post-activation to transform the input data to some reasonable form
x = Conv2D(kernel_size=3, filters=16, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(input_tensor)
x = BatchNormalization()(x)
x = Activation(relu)(x)

# F_1
x = block(16)(x)
# F_2
x = block(16)(x)

# F_3
# H_3 is the function from the tensor of size 28x28x16 to the the tensor of size 28x28x32
# and we can't add together tensors of inconsistent sizes, so we use upscale=True
# x = block(32, upscale=True)(x)       # !!! <------- Uncomment for local evaluation
# F_4
# x = block(32)(x)                     # !!! <------- Uncomment for local evaluation
# F_5
# x = block(32)(x)                     # !!! <------- Uncomment for local evaluation

# F_6
# x = block(48, upscale=True)(x)       # !!! <------- Uncomment for local evaluation
# F_7
# x = block(48)(x)                     # !!! <------- Uncomment for local evaluation

# last activation of the entire network's output
x = BatchNormalization()(x)
x = Activation(relu)(x)

# average pooling across the channels
# 28x28x48 -> 1x48
x = GlobalAveragePooling2D()(x)

# dropout for more robust learning
x = Dropout(0.2)(x)

# last softmax layer
x = Dense(units=10, kernel_regularizer=regularizers.l2(0.01))(x)
x = Activation(softmax)(x)

model = Model(inputs=input_tensor, outputs=x)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# The depth of the network is 16 layers.

# Let's load and preprocess the data to test our network's possibilities.

# In[ ]:


df_train = pd.read_csv('../input/train.csv')

y_train_ = df_train.ix[:, 0].values.astype(np.int).reshape(-1, 1)
x_train = df_train.ix[:, 1:].values.astype(np.float32).reshape((-1, 28, 28, 1))


# In[ ]:


df_test = pd.read_csv('../input/test.csv')

x_test = df_test.values.astype(np.float32).reshape((-1, 28, 28, 1))


# In[ ]:


y_train = OneHotEncoder(sparse=False).fit_transform(y_train_)


# Randomly take 20% of data for validation:

# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train_)


# Normalize data by subtracting the mean image:

# In[ ]:


m = x_train.mean(axis=0)

x_train -= m
x_val -= m
x_test -= m


# Now we will define two useful callbacks: one for the model checkpointing and one for managing the learning rate policy.

# In[ ]:


from keras.callbacks import LearningRateScheduler, ModelCheckpoint


# In[ ]:


mc = ModelCheckpoint('weights.best.keras', monitor='val_acc', save_best_only=True)


# Let's use the sigmoidal decay as the learning rate policy:

# In[ ]:


def sigmoidal_decay(e, start=0, end=100, lr_start=1e-3, lr_end=1e-5):
    if e < start:
        return lr_start
    
    if e > end:
        return lr_end
    
    middle = (start + end) / 2
    s = lambda x: 1 / (1 + np.exp(-x))
    
    return s(13 * (-e + middle) / np.abs(end - start)) * np.abs(lr_start - lr_end) + lr_end


# In[ ]:


xs = np.linspace(0, 100)
ys = np.vectorize(sigmoidal_decay)(xs)
plt.plot(xs, ys)
plt.show()


# In[ ]:


EPOCHS = 3                        # !!! <------- Chnage to 30-100 for local evaluation


# In[ ]:


lr = LearningRateScheduler(lambda e: sigmoidal_decay(e, end=EPOCHS))


# And now we can train the model:

# In[ ]:


hist = model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_val, y_val), batch_size=512, callbacks=[lr, mc])


# Training the full model for 100 epochs leads to 99.17% validation and 99.314% test accuracy. One epoch takes 40s on GTX 960.

# In[ ]:


loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs = np.arange(1, EPOCHS + 1)

plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.show()


# In[ ]:


acc = hist.history['acc']
val_acc = hist.history['val_acc']
epochs = np.arange(1, EPOCHS + 1)

plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.show()


# Now we can predict the test values and save them.

# In[ ]:


model.load_weights('weights.best.keras')


# In[ ]:


p_test = model.predict(x_test, batch_size=512)
p_test = np.argmax(p_test, axis=1)


# In[ ]:


pd.DataFrame({'ImageId': 1 + np.arange(p_test.shape[0]), 'Label': p_test}).to_csv('output.csv', index=False)


# Now we can predict the test values and save them.

# In[ ]:


model.load_weights('weights.best.keras')


# In[ ]:


p_test = model.predict(x_test, batch_size=512)
p_test = np.argmax(p_test, axis=1)


# In[ ]:


pd.DataFrame({'ImageId': 1 + np.arange(p_test.shape[0]), 'Label': p_test}).to_csv('output.csv', index=False)

