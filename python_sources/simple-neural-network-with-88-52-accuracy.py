#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Neural Network Classifier
# 
# The data of this project comes from Kaggle.com(https://www.kaggle.com/ronitf/heart-disease-uci/kernels).
# The originl data is from UCI's Macine Learning Repository (https://archive.ics.uci.edu/ml/datasets/Heart+Disease).
# 
# With data of heart diease patient, we can build a neural network model and train the model with training set to make precise predictions.

# In[69]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization
import matplotlib.pyplot as plt


# First we use pandas.read_csv() to read the data, check the size and take a glance. The data is not very large so a complex neural network is very likely to overfit. To reach a higher accuracy and F-score we need to choose architecture carefully.

# In[70]:


data = pd.read_csv('../input/heart.csv')


# In[71]:


data.shape


# In[72]:


data.head()


# According to the information of data, most of variables are catagorial. We will then process those features with pd.get_dummies().

# In[73]:


catagorialList = ['sex','cp','fbs','restecg','exang','ca','thal']
for item in catagorialList:
    data[item] = data[item].astype('object')


# In[74]:


data.dtypes


# In[75]:


data = pd.get_dummies(data, drop_first=True)


# In[76]:


data.head()


# We will normalize the input features to avoid any one feature to dominate the training of neural network.
# The factor we use in normalization will be kept to make predictions.

# In[77]:


y = data['target'].values
y = y.reshape(y.shape[0],1)
x = data.drop(['target'],axis=1)
minx = np.min(x)
maxx = np.max(x)
x = (x - minx) / (maxx - minx)
x.head()


# Since the dataset is not large, we will not use cross-validation set to let neural network model have enough data to train.

# In[78]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)


# In keras, It is quite convenient to setup a neural network model.We will set 21 neurons as input layer, 12 neurons in the hidden layer and an output layer.

# In[79]:


model = Sequential()
model.add(Dense(12, input_dim=21, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))


# In[80]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[81]:


output = model.fit(x_train, y_train,validation_split=0.2, epochs=200, batch_size=x_train.shape[0]//2)


# In[82]:


scores = model.evaluate(x_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# We reached an accuracy of 88.52%.
# 
# During the tuning of hyperparameters, the sigmoid activation seems to have a more stable accuracy than Relu function. This is mainly becaus Relu don't learn from negative prediction and learn faster in a positive prediction, thus making the model easier to overfit.
# 
# Here is the summary of accuracy and loss during training from https://keras.io/visualization/.

# In[83]:


plt.plot(output.history['acc'])
plt.plot(output.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Accuracy.png',dpi=100)
plt.show()


# In[84]:


plt.plot(output.history['loss'])
plt.plot(output.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Loss.png',dpi=100)
plt.show()


# In[ ]:




