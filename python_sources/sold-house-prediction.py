#!/usr/bin/env python
# coding: utf-8

# The variable in the 19th column 'Sold' is the output variable to be predicted. All other variables are to be used as the predictor variables.
# We are going to predict if the House can be sold or Not.

# # 1. Importing Libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Spliting data and creating model libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential #initialize neural network library
from keras.layers import Dense #build our layers library
from tensorflow import keras
from keras.models import Sequential

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # 2. loading the Dataset

# In[ ]:


data_train = pd.read_csv("../input/house-price2/House-Price2.csv")
data_train.head()


# # 3. Exploratory Data Analysis

# In[ ]:


data_train.info()


# The section code in above, You can find some information. 
# There are 19 features in the House Price data. 
# It has information about 506 Houses. 
# 2 features are integer type
# 3 features are object type
# 14 features are float type. 
#  
# Some features contain missing value.

# In[ ]:


# We can observe number of sold House
sns.countplot(data_train["Sold"])
plt.show()


# In[ ]:


data_train.describe()


# In[ ]:


# Let's plot the distribution of Hot Room and sold
sns.jointplot(x='n_hot_rooms', y='Sold', data=data_train)


# Identifying Categorical Variables

# We have 3 Categorical variables : airport, Waterbody and bus terminal

# In[ ]:


sns.countplot(x='airport', data=data_train)


# In[ ]:


sns.countplot(x='waterbody', data=data_train)


# In[ ]:


sns.countplot(x='bus_ter', data=data_train)


# Outlier Treatment

# In[ ]:


data_train.info()


# In[ ]:


np.percentile(data_train.n_hot_rooms,[99])


# In[ ]:


np.percentile(data_train.n_hot_rooms,[99])[0]


# In[ ]:


nv = np.percentile(data_train.n_hot_rooms,[99])[0]


# In[ ]:


data_train[(data_train.n_hot_rooms > nv)]


# We got all the values of n_hot_rooms greater than percentile value.
# 
# replace the value of 101.12 and 81.12 by values close to 15.399519999999

# In[ ]:


data_train.n_hot_rooms[(data_train.n_hot_rooms > 3 * nv)] = 3 * nv


# In[ ]:


data_train[(data_train.n_hot_rooms > nv)]


# In[ ]:


np.percentile(data_train.rainfall,[1])[0]


# In[ ]:


lv = np.percentile(data_train.rainfall,[1])[0]


# In[ ]:


data_train[(data_train.rainfall < lv)]


# In[ ]:


data_train[(data_train.rainfall < lv)]


# # 4. Preparing Data

# Missing values Imputation

# In[ ]:


data_train.info()


# We have missing values in n_hos_beds variable : 498 of 506

# In[ ]:


#Impute Missing values for 1 columns
data_train.n_hos_beds = data_train.n_hos_beds.fillna(data_train.n_hos_beds.mean())
# For all columns : df = df.fillna(df.mean())


# In[ ]:


data_train.info()


# Variable Transformation and deletion

# In[ ]:


data_train.head()


# We are going to create an average variable for dist1, dist2, dist3 and dist4

# In[ ]:


data_train['avg_dist'] = (data_train.dist1 + data_train.dist2 + data_train.dist3 + data_train.dist4) / 4


# In[ ]:


data_train.describe()


# Now we can delete the variable dist1, dist2, dist3 and dist4

# In[ ]:


del data_train['dist1']


# In[ ]:


del data_train['dist2']


# In[ ]:


del data_train['dist3']


# In[ ]:


del data_train['dist4']


# In[ ]:


data_train.head()


# In[ ]:


data_train.shape


# In[ ]:


del data_train['bus_ter']


# Dummy variable creation

# We are going to create dummy variable for categorical variables

# In[ ]:


data_train = pd.get_dummies(data_train)


# In[ ]:


data_train.head()


# In[ ]:


data_train.shape


# In[ ]:


del data_train['airport_NO']


# In[ ]:


del data_train['waterbody_None']


# In[ ]:


data_train.head()


# In[ ]:


data_train.shape


# Correlation Analysis

# In[ ]:


data_train.corr()


# In[ ]:


data_test = data_train


# In[ ]:


data_test.shape


# In[ ]:





# # 5. Train and Test Split

# Our train data splitted as train and test data in order to feed in a Neural  Network correctly.
# Train data is 80% and test data 20% of the House Prices Dataset.

# In[ ]:


X = data_train.drop(["Sold"],axis=1)
Y = data_train["Sold"]
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
print("x_train shape: ",x_train.shape)
print("y_train shape: ",y_train.shape)
print("x_test shape: ",x_test.shape)
print("y_test shape: ",y_test.shape)


# In[ ]:


X.shape


# In[ ]:





# # 6. Neural Network : Keras Model

# In[ ]:


my_model = Sequential() # initialize neural network
my_model.add(Dense(units = 128, activation = 'relu', input_dim = X.shape[1]))
my_model.add(Dense(units = 32, activation = 'relu'))
my_model.add(Dense(units = 16, activation = 'relu'))
my_model.add(Dense(units = 8, activation = 'relu'))
my_model.add(Dense(units = 4, activation = 'relu'))
my_model.add(Dense(units = 1, activation = 'sigmoid')) #output layer
my_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[ ]:


my_model.summary()


# In[ ]:


keras.utils.plot_model(my_model)


# In[ ]:


model = my_model.fit(x_train,y_train,epochs=750)
mean = np.mean(model.history['accuracy'])
print("Accuracy mean: "+ str(mean))


# Let's see the parameters for our trained model

# In[ ]:


model.params


# In[ ]:


pd.DataFrame(model.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()


# Model Evaluation using Confusion Matrix

# In[ ]:


y_predict = my_model.predict(X)
cm = confusion_matrix(Y,np.argmax(y_predict, axis=1))

f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, ax=ax)


# In[ ]:


ids = data_test['House_id']
#predict = classifier.predict(data_test_x)
predict = my_model.predict(X)

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'House_id' : ids, 'Sold': np.argmax(predict,axis=1)})
output.to_csv('submission.csv', index=False)


# In[ ]:


submission = pd.read_csv('submission.csv', index_col=0)
submission.head(50)

