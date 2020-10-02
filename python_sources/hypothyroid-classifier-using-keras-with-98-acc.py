#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# #### Import keras to our build classifier for Hypothyroid data

# In[2]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# #### Load dataset from csv using pandas

# In[3]:


dataset = pd.read_csv('../input/hypothyroid.csv')
dataset.head()


# #### We now replace '?' values in dataset with NaN values so that it becomes easy to count them

# In[4]:


data_copy = dataset.copy(deep = True)
data_copy.replace(to_replace='?', inplace=True, value=np.NaN)
print(data_copy.isnull().sum())


# #### Then replace/encode categorical values using sklearn's LabelEncoder() class or panda's replace() method

# In[5]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data_copy['Unnamed: 0'] = encoder.fit_transform(data_copy['Unnamed: 0'])
data_copy['Sex'] = data_copy['Sex'].replace({'M':0, 'F':1})
data_copy = data_copy.replace(to_replace={'f':0,'t':1, 'y':1, 'n':0})
print(data_copy.head())


# #### Convert dtype='object' to 'int64' or 'float64' to view the statistics later

# In[6]:


# Columns with dtype as 'object'
cols = data_copy.columns[data_copy.dtypes.eq('object')]
# Convert to numeric values
data_copy[cols] = data_copy[cols].apply(pd.to_numeric, errors='coerce')
data_copy.info()


# #### Plot the statistics column wise to visualize the patterns or trends in data

# In[7]:


p = data_copy.hist(figsize = (20,20))


# #### From the statistics plotted above, we can then replace the NaN values with appropriate values

# In[8]:


data_copy['Age'].fillna(data_copy['Age'].mean(), inplace = True)
data_copy['Sex'].fillna(0, inplace = True)
data_copy['TSH'].fillna(data_copy['TSH'].mean(), inplace = True)
data_copy['T3'].fillna(data_copy['T3'].median(), inplace = True)
data_copy['TT4'].fillna(data_copy['TT4'].median(), inplace = True)
data_copy['FTI'].fillna(data_copy['FTI'].median(), inplace = True)
data_copy['T4U'].fillna(data_copy['T4U'].mean(), inplace = True)
data_copy['TBG'].fillna(data_copy['TBG'].mean(), inplace = True)


# #### Visualize the imputed data

# In[9]:


p = data_copy.hist(figsize = (20,20))


# #### Now again check the number of NaN values for each column

# In[10]:


data_copy.isnull().sum()


# #### Here, we separate the features and target attributes into X and Y respectively

# In[11]:


X = data_copy.iloc[:,1:]
Y = data_copy.iloc[:, 0]
print(X.shape, Y.shape)


# #### Now, we split the dataset into train and test data. I have specified test_size as 20% and also randomness factor

# In[12]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
print(X_train.shape, Y_train.shape)


# #### We now scale the train and test data using sklearn's StandardScaler class

# In[13]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# #### I am saving scaler object to file so that it can later used to scale the test data for the saved model

# In[14]:


import pickle
with open('scaler.obj', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)


# To load the object back from file, we can use scaler = pickle.load(scaler_file)

# #### Here, now we define a sequential model with one hidden layer and one output layer. Here, input_dim=25 as we have 25 features

# In[15]:


model = Sequential()
model.add(Dense(64, kernel_initializer='uniform', input_dim=25, activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))


# #### Now, compile the model with 'adam' optimizer and loss function as 'binary_crossentropy'

# In[16]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# #### We call fit() on training data with validation split=0.2 for 50 iterations

# In[17]:


history = model.fit(X_train, Y_train, epochs=50, validation_split=0.2, batch_size=40,  verbose=2)


# #### Now, as the model is trained, we can evaluate it on test data. Yay! We got a very good accuracy of 98%

# In[18]:


scores = model.evaluate(X_test, Y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# #### We can now plot training and validation losses using matplotlib and visualize that the losses have converged at a very good rate

# In[19]:


import matplotlib.pyplot as plt

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# #### We now save the trained model

# In[20]:


model.save('model.h5')

