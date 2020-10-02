#!/usr/bin/env python
# coding: utf-8

# In[98]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Reading databases

# In[71]:


wines = pd.read_csv("../input/wine_dataset.csv")


# ## Inspecting general information about the dataset

# In[72]:


print(wines.head(15))
print(wines.shape)
print(wines.info())


# We have 6497 records with 13 variables available. The feature Style allows to classify a wine as red o white. For any variables there are not missing values. Good stuff!

# Let's plot the proportion for both clases

# In[73]:


data = wines["style"].value_counts(normalize=True)
data


# In[74]:


data = wines["style"].value_counts(normalize=True)
data.plot(kind='bar',figsize=(15,6))
plt.title("Wine class proportion",fontsize= 16)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()


# ## Basic exploratory analysis

# In[75]:


wines.describe()


# In[76]:


# Compute the correlation matrix
corr=wines.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))


# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[77]:


# lets get some grouped stats for both clasees
wines.groupby(['style']).mean()


# It seems that in average, red wine usually is less acid than white wine (the opposite is true when we look at `volatile_acidity` feature). 
# The 'residual_sugar' in average is almost 3 times bigger for a white wine.
# The feature 'free_sulfur_dioxide' is highly and positive correlated with `total_sulfur_dioxide', which kinf of makes sense.
# 

# ## We will scale our continuos features

# In[78]:


#Scaling the continuos variables
wines_scale = wines.copy()
scaler = preprocessing.StandardScaler()
columns = wines.columns[0:12]
wines_scale[columns] = scaler.fit_transform(wines_scale[columns])
wines_scale.head()


# In[79]:


sns.boxplot(x="style", y="residual_sugar", data=wines, palette="Set2" )


# In[80]:


sns.boxplot(x="style", y="total_sulfur_dioxide", data=wines, palette="Set2" )


# We confirm the facts seen after grouping by classes and calculating average stats for the features. Also, we can see a few outliers.

# ## Encoding the output vector

# In[102]:


label_encoder = preprocessing.LabelEncoder()
le = label_encoder.fit_transform(wines_scale['style'])
print(le)
print(wines_scale.info())
print(wines_scale.head(5))


# ## Spliting training and test datasets

# In[103]:


X_train, X_test, y_train, y_test=train_test_split(wines_scale.iloc[:,0:12], le, test_size=0.33, random_state=8)


# In[95]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ## Now we create a neural network model

# In[100]:


# Initialize the constructor
model=Sequential()
# Add an input layer 
model.add(Dense(13, activation='relu', input_shape=(12,)))

# Add one hidden layer 
model.add(Dense(60, activation='relu'))
model.add(Dense(30, activation='relu'))

# Add an output layer 
model.add(Dense(1, activation='relu'))

model.output_shape
# Model summary
model.summary()


# In[104]:


model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])
model.fit(X_train, y_train,epochs=400, batch_size=100, verbose=0, callbacks=[EarlyStopping(monitor='acc', patience=200)])


# ## Now lets evaluate the model

# In[106]:


# over the training set
y_pred=model.predict_classes(X_train)
print(confusion_matrix(y_train, y_pred))
print(classification_report(y_train, y_pred))


# In[107]:


# over the test set
y_pred=model.predict_classes(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# The neural network proposed has a high predictive power
