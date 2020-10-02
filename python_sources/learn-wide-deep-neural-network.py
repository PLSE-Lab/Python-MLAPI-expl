#!/usr/bin/env python
# coding: utf-8

# > Please read this : I have not included much explaination in this code because they are preety easy to understand. If you are finding anything complex, please let me know i will explain that.
# 
# # Introduction
# 
# Topics Covered :-
# 
# 1. Functional API (Concat Layers)
# 2. Multiple Inputs
# 3. Multiple Outputs
# 
# > I am using Keras and Tensorflow to build these models.
# 
# ![](https://2.bp.blogspot.com/-wkrmRibw_GM/V3Mg3O3Q0-I/AAAAAAAABG0/Jm3Nl4-VcYIJ44dA5nSz6vpTyCKF2KWQgCKgB/s640/image03.png)
# 
# Read more about [Wide Deep Neural Networks](https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html)

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[ ]:


housing = fetch_california_housing()
type(housing)


# In[ ]:


print(housing['DESCR'])


# > Let us make train and dev set to build our model and I will use test set to evaluate at end.

# In[ ]:


X_train_full,X_test,y_train_full,y_test = train_test_split(housing.data,housing.target)
X_train,X_valid,y_train,y_valid = train_test_split(X_train_full,y_train_full)


# In[ ]:


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


# # Functional API
# 
# ## Concat Layers
# 
# The Keras functional API is a way to create models that is more flexible than the tf.keras.Sequential API. The functional API can handle models with non-linear topology, models with shared layers, and models with multiple inputs or outputs.
# 
# The main idea that a deep learning model is usually a directed acyclic graph (DAG) of layers. So the functional API is a way to build graphs of layers.

# In[ ]:


from tensorflow import keras

input_ = keras.layers.Input(shape=X_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation="relu")(input_)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.models.Model(inputs=[input_], outputs=[output])


# In[ ]:


model.summary()


# In[ ]:


keras.utils.plot_model(model,'wide_alpha.png',show_shapes=True)


# In[ ]:


model.compile(loss = 'mean_squared_error',optimizer=keras.optimizers.SGD(lr = 1e-3))


# In[ ]:


history = model.fit(X_train,y_train,
          epochs = 50,
          validation_data = (X_valid,y_valid))


# In[ ]:


fig = plt.figure(dpi = 100,figsize = (5,3))
ax = fig.add_axes([1,1,1,1])
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(loss)+1)
ax.plot(epochs,loss,lw = 2,color = 'green',label = 'Loss')
ax.plot(epochs,val_loss,lw = 2,color = 'blue',label = 'Val Loss')
plt.grid(True)
plt.legend()
ax.set(xlabel = 'Number of Epochs',ylabel = 'Loss',title = 'Loss Curve')
plt.show()


# > Now let us predict outcome for first three observation in our test set.

# In[ ]:


X_new = X_test[:3]
y_pred = model.predict(X_new)
y_pred


# # Multiple Inputs
# 
# Network with multiple inputs, I will overlap some inputs. I will pass 5 features directly to output layer and 6 features to Deep Networks.

# In[ ]:


input_A = keras.layers.Input(shape = [5],name = 'wide_input')
input_B = keras.layers.Input(shape = [6],name = 'deep_input')

hidden1 = keras.layers.Dense(30,activation = 'relu')(input_B)
hidden2 = keras.layers.Dense(30,activation='relu')(hidden1)

concat = keras.layers.concatenate([input_A,hidden2])

output = keras.layers.Dense(1)(concat)

model = keras.Model(inputs = [input_A,input_B],outputs = [output])


# In[ ]:


model.summary()


# In[ ]:


keras.utils.plot_model(model,'multi.png',show_shapes=True)


# In[ ]:


model.compile(loss = 'mean_squared_error',optimizer=keras.optimizers.SGD(lr = 1e-3))


# > As we are passing two inputs, We need two input set. Let us make that.

# In[ ]:


X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]


# In[ ]:


history = model.fit((X_train_A,X_train_B),y_train,
          epochs = 50,
          validation_data = ((X_valid_A,X_valid_B),y_valid))


# In[ ]:


fig = plt.figure(dpi = 100,figsize = (5,3))
ax = fig.add_axes([1,1,1,1])
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(loss)+1)
ax.plot(epochs,loss,lw = 2,color = 'green',label = 'Loss')
ax.plot(epochs,val_loss,lw = 2,color = 'blue',label = 'Val Loss')
plt.grid(True)
plt.legend()
ax.set(xlabel = 'Number of Epochs',ylabel = 'Loss',title = 'Loss Curve')
plt.show()


# In[ ]:


y_pred = model.predict((X_new_A,X_new_B))
y_pred


# # Multiple Outputs
# 
# Now build a model with multiple inputs as well as multiple outputs.

# In[ ]:


input_A = keras.layers.Input(shape = [5],name = 'wide_input')
input_B = keras.layers.Input(shape = [6],name = 'deep_input')

hidden1 = keras.layers.Dense(30,activation='relu')(input_B)
hidden2 = keras.layers.Dense(30,activation='relu')(hidden1)

concat = keras.layers.concatenate([input_A,hidden2])

output = keras.layers.Dense(1,name = 'main_output')(concat)
aux_output = keras.layers.Dense(1,name = 'aux_output')(hidden2)

model = keras.Model(inputs = [input_A,input_B],outputs = [output,aux_output])

model.summary()


# In[ ]:


keras.utils.plot_model(model,'complex.png',show_shapes=True)


# In[ ]:


model.compile(loss = ['mse','mse'],loss_weights = [0.9,0.1],optimizer='sgd')


# > I am using same target at both outputs , So i am passing y_train multiple times.

# In[ ]:


history = model.fit((X_train_A,X_train_B),(y_train,y_train),
          epochs = 50,
          validation_data = ((X_valid_A,X_valid_B),(y_valid,y_valid)))


# In[ ]:


pd.DataFrame(history.history).head()


# In[ ]:


fig = plt.figure(dpi = 100,figsize = (5,3))
ax = fig.add_axes([1,1,1,1])
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(loss)+1)
ax.plot(epochs,loss,lw = 2,color = 'green',label = 'Loss')
ax.plot(epochs,val_loss,lw = 2,color = 'blue',label = 'Val Loss')
plt.grid(True)
plt.legend()
ax.set(xlabel = 'Number of Epochs',ylabel = 'Loss',title = 'Loss Curve')
plt.show()


# > Evaluate returns total_loss, main_loss and aux_loss.

# In[ ]:


total_loss,main_loss,aux_loss = model.evaluate((X_test_A,X_test_B),
                                               (y_test,y_test))


# > We can also predict at each output.

# In[ ]:


y_pred_main,y_pred_aux = model.predict([X_new_A,X_new_B])


# In[ ]:


y_pred_main


# In[ ]:


y_pred_aux


# > Thank you for reading my notebook. Let me know how you find this useful. Happy Learning !!
