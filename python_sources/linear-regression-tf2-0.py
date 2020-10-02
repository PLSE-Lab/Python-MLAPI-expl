#!/usr/bin/env python
# coding: utf-8

# # Import relevant libraries 

# In[ ]:


import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt


# # Creating dataset and saving it

# In[ ]:


observations=10000
xs=np.random.uniform(-10,20,observations)
zs=np.random.uniform(-20,25,observations)
inputs=np.column_stack((xs,zs))
noise=np.random.uniform(-1,1,observations)
targets=2*xs-3*zs+4+noise
targets=targets.reshape(10000,1).round(1)
np.savez('data_set',input=inputs,target=targets)


# # Loading dataset file

# In[ ]:


data_set=np.load('data_set.npz')
data_set['target'].shape


# # Creating, customizing & training the model

# In[ ]:


input_size=2
output_size=1

model=tf.keras.Sequential([
    tf.keras.layers.Dense(output_size,
                         kernel_initializer=tf.random_uniform_initializer(-0.1,0.1),
                         bias_initializer=tf.random_uniform_initializer(-0.1,0.1),
                         )
])

custom_optimizer=tf.keras.optimizers.SGD(learning_rate=0.002)
model.compile(custom_optimizer,'mean_squared_error')
model.fit(data_set['input'],data_set['target'],epochs=10,verbose=2)


# In[ ]:


weights=model.layers[0].get_weights()[0]
biases=model.layers[0].get_weights()[1]
weights


# In[ ]:


biases


# # Printing what our model predicts and comparing it with calculated targets

# In[ ]:


predict=np.array(model.predict_on_batch(data_set['input']))
predict.round(1)


# # Printing Targets

# In[ ]:


data_set['target']


# In[ ]:


plt.plot(np.squeeze(model.predict_on_batch(data_set['input'])),np.squeeze(data_set['target']))
plt.xlabel('Inputs')
plt.ylabel('Targets')
plt.show()

