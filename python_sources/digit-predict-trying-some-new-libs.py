#!/usr/bin/env python
# coding: utf-8

# Hello, 
# 
# I thought to try something new, as in a new library that some lovely people I know put together. You can find more info [here](http://medium.com/dataseries/localized-computation-3e1edad63000).
# 
# Now let's have a quick ride:

# In[ ]:


pip install larq


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import larq as lq 


# In[ ]:


train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')


# In[ ]:


x_train = (train.iloc[:,1:].values).astype('float32') # all pixel values
y_train = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits


# In[ ]:


# you can on hot encode or us it as it is, only change the loss to sparse_categorical_crossentropy
from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)


# In[ ]:


# don't forget test set
test = test.values.astype('float32')


# In[ ]:


train_images, test_images, train_labels, test_labels = train_test_split(
                    x_train, y_train, 
                    test_size=0.33, random_state=42)


# In[ ]:


print(train_images.shape)
print(test_images.shape)


# In[ ]:


train_images = train_images.reshape((28140, 28, 28, 1))  
test_images = test_images.reshape((13860, 28, 28, 1)) 

test = test.reshape((28000, 28, 28, 1))


# In[ ]:


# Normalize pixel values to be between -1 and 1
train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1


# In[ ]:


kwargs = dict(input_quantizer="ste_sign", kernel_quantizer="ste_sign", kernel_constraint="weight_clip")

model = tf.keras.models.Sequential()

model.add(lq.layers.QuantConv2D(32, (3, 3), kernel_quantizer="ste_sign", 
                                kernel_constraint="weight_clip", use_bias=False, 
                                input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.BatchNormalization(scale=False))

model.add(lq.layers.QuantConv2D(64, (3, 3), use_bias=False, **kwargs))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.BatchNormalization(scale=False))

model.add(lq.layers.QuantConv2D(64, (3, 3), use_bias=False, **kwargs))
model.add(tf.keras.layers.BatchNormalization(scale=False))
model.add(tf.keras.layers.Flatten())

model.add(lq.layers.QuantDense(64, use_bias=False, **kwargs))
model.add(tf.keras.layers.BatchNormalization(scale=False))
model.add(lq.layers.QuantDense(10, use_bias=False, **kwargs))
model.add(tf.keras.layers.BatchNormalization(scale=False))
model.add(tf.keras.layers.Activation("softmax"))


# In[ ]:


# always nice to have a look at this lovely model
lq.models.summary(model)


# In[ ]:


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
sparse_categorical_crossentropy is the loss function for integer labels

h = model.fit(train_images, train_labels, batch_size=64, epochs=6)

test_loss, test_acc = model.evaluate(test_images, test_labels)


# Epoch 1/6
# 28140/28140 [==============================] - 41s 1ms/step - loss: 0.7582 - acc: 0.8733
# Epoch 2/6
# 28140/28140 [==============================] - 39s 1ms/step - loss: 0.5240 - acc: 0.9502
# Epoch 3/6
# 28140/28140 [==============================] - 40s 1ms/step - loss: 0.4837 - acc: 0.9603
# Epoch 4/6
# 28140/28140 [==============================] - 36s 1ms/step - loss: 0.4613 - acc: 0.9644
# Epoch 5/6
# 28140/28140 [==============================] - 37s 1ms/step - loss: 0.4494 - acc: 0.9682
# Epoch 6/6
# 28140/28140 [==============================] - 36s 1ms/step - loss: 0.4420 - acc: 0.9709
# 13860/13860 [==============================] - 5s 343us/step

# In[ ]:


print(f"Test accuracy {test_acc * 100:.2f} %")
print(f"Test loss {test_loss}")


# Test accuracy 97.80 %
# Test loss 0.405355900223293

# In[ ]:


pd.DataFrame(h.history).plot()
# and as an ending, I'll let you plot the accuracy and loss changes per epoch ;) 


# 

# 

# In[ ]:





# In[ ]:





# In[ ]:




