#!/usr/bin/env python
# coding: utf-8

# ### Import TensorFlow

# In[ ]:


import tensorflow as tf
print('Using TensorFlow version', tf.__version__)


# ### Import MNIST

# In[ ]:


from tensorflow.keras.datasets import mnist
(x_train , y_train) , (x_test , y_test) = mnist.load_data()


# ### Shapes of Imported Arrays

# In[ ]:


print('x_train shape:' , x_train.shape)
print('y_train shape:' , y_train.shape)
print('x_test shape:' , x_test.shape)
print('y_test shape:' , y_test.shape)


# ### Plot an Image Example

# In[ ]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.imshow(x_train[0] , cmap = 'binary')
plt.show()


# ### Display Labels

# In[ ]:


y_train[0]


# In[ ]:


print(set(y_train))


# # One Hot Encoding
# After this encoding, every label will be converted to a list with 10 elements and the element at index to the corresponding class will be set to 1, rest will be set to 0:
# 
# | original label | one-hot encoded label |
# |------|------|
# | 5 | [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] |
# | 7 | [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] |
# | 1 | [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] |
# 
# ### Encoding Labels

# In[ ]:


from tensorflow.keras.utils import to_categorical

y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)


# ### Validated Shapes

# In[ ]:


print('y_train_encoded:' , y_train_encoded.shape)
print('y_test_encoded:' , y_test_encoded.shape)


# ### Display Encoded Labels

# In[ ]:


y_train_encoded[0]


# ### Unrolling N-dimensional Arrays to Vectors

# In[ ]:


import numpy as np
x_train_reshaped = np.reshape(x_train , (60000 , 784))
x_test_reshaped = np.reshape(x_test, (10000 , 784))
print('x_test_reshaped shape:' , x_train_reshaped.shape)
print('x_train_reshaped:' , x_test_reshaped.shape)


# ### Display Pixel Values

# In[ ]:


print(set(x_train_reshaped[0]))


# ### Data Normalization

# In[ ]:


x_mean = np.mean(x_train_reshaped)
x_std = np.std(x_train_reshaped)

epilson = 1e-10

x_train_norm = (x_train_reshaped - x_mean)/(x_std + epilson)
x_test_norm = (x_test_reshaped - x_mean) / (x_std + epilson)


# ### Display Normalized Pixel Values

# In[ ]:


print(set(x_train_norm[0]))


# ### Creating the Model

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(128 , activation = 'relu' , input_shape = (784 , )),
    Dense(128 , activation = 'relu'),
    Dense(10 , activation = 'softmax')
])


# ### Compiling the Model

# In[ ]:


model.compile(
    optimizer = 'sgd',
    loss = 'categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()


# ### Training the Model

# In[ ]:


model.fit(x_train_norm , y_train_encoded , epochs=3)


# ### Evaluating the Model

# In[ ]:


loss , accuracy = model.evaluate (x_test_norm , y_test_encoded)
print('Test set accuracy' , accuracy*100)


# ### Predictions on Test Set

# In[ ]:


preds = model.predict(x_test_norm)
print('Shape of pred:' , preds.shape)


# ### Plotting the Results

# In[ ]:


plt.figure(figsize = (12 , 12))

start_index = 0

for i in range(15):
    plt.subplot( 5 , 5 , i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    pred = np.argmax(preds[start_index+i])
    gt = y_test[start_index+i]
    
    col = 'g'
    if pred != gt:
        col = 'r'
        
    plt.xlabel('i ={} , pred ={} , gt={}'.format(start_index+i , pred , gt) , color= col)
    plt.imshow(x_test[start_index+i] , cmap = 'binary')
plt.show()


# In[ ]:


plt.plot(preds[8])
plt.show()

