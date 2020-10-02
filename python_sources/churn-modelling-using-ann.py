#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ### Importing the Required Packages to Solve the Problem

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Import the Required Dataset

# In[ ]:


data = pd.read_csv("../input/churn-modelling/Churn_Modelling.csv")


# ### Data Exploration

# In[ ]:


data.head()


# In[ ]:


data.shape


# This data shows that there are 10,000 recors with 14 Parameters. Out of 14, 13 parameters are predictor variables and 1 parameter is a response variable ("Exited").

# In[ ]:


data.info()


# There are 2 float type parameters, 9 Integer variables and 3 String type variables.

# ### Identify the Missing Values

# In[ ]:


data.isnull().sum()


# Fortunately, there are no missing values in our data

# ### Split the Data into Predictor and Response Variables

# In[ ]:


x = data.drop("Exited", axis=1)


# In[ ]:


y = data['Exited']


# In[ ]:


x.shape


# In[ ]:


y.shape


# In[ ]:


data['Surname'].value_counts()


# In[ ]:


data['Geography'].value_counts()


# In[ ]:


sns.catplot('Geography', data=data, kind='count')
plt.xlabel("Country")
plt.ylabel("Number of People")
plt.title("Number of People for Each country")


# Based on Geography, we identified that there are people from France,Germany and Spain countries in that Bank.

# In[ ]:


data['Gender'].value_counts()


# In[ ]:


sns.catplot('Gender', data=data, kind='count')
plt.xlabel("Gender Category")
plt.ylabel("Number of People")
plt.title("Gender Classification")


# Graph tells us that the Bank has 5457 Male customers and 4543 female customers

# In[ ]:


data['HasCrCard'].value_counts()


# In[ ]:


sns.catplot('HasCrCard', data=data, kind='count')
plt.xlabel("Category")
plt.ylabel("Number of People")
plt.title("Identify People with Credit Card")


# Graph tells us that there are more people with credit card and also few people without credit cards. Number of people with credit cards are 7055 and people without credit cards are 2945

# In[ ]:


data['IsActiveMember'].value_counts()


# In[ ]:


sns.catplot('IsActiveMember', data=data, kind='count')
plt.xlabel("Category")
plt.ylabel("Number of People")
plt.title("Identify Active Members")


# Graph shows us that active customers are 5151 and non-active customers are almost as equal as active customers with the value 4849.

# Since there are 3 object type data variables(i.e, Surname, Geography and Gender). Out of which, surname is not required for our data processing, so, we neglect it and we create some dummy variables for Geography and Gender columns for performing the computations on string type data.

# In[ ]:


geography = pd.get_dummies(x['Geography'], drop_first=True)
gender = pd.get_dummies(x['Gender'], drop_first=True)


# In[ ]:


geography.shape


# In[ ]:


gender.shape


# In[ ]:


x.shape


# In[ ]:


x = pd.concat([x, geography, gender], axis=1)


# In[ ]:


x.shape


# In[ ]:


x.info()


# In[ ]:


x=x.drop(["Geography", "Gender", "Surname","RowNumber","CustomerId"], axis=1)


# In[ ]:


x.shape


# As you have added the encoded values for both geography and gender parameters to your predictor variables dataset, we need to remove the object columns.

# ### Split the Dataset into train and test sets
# 
# Test size is taken as 25%.

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=50)


# In[ ]:


x_train.shape


# In[ ]:


x_test.shape


# In[ ]:


y_train.shape


# In[ ]:


y_test.shape


# In[ ]:


x_train.head()


# ### Perform Feature Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scale = StandardScaler()


# In[ ]:


x_train = scale.fit_transform(x_train)


# In[ ]:


x_test = scale.fit_transform(x_test)


# In[ ]:


x_train.shape


# ### Model implementation

# Importing all the Deep Learning required Libraries

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU, PReLU, ELU
from keras.layers import Dropout


# ### Initializing the ANN Model

# In[ ]:


model = Sequential()


# Adding the input layer and the hidden layers

# In[ ]:


model.add(Dense(units = 10, kernel_initializer = "he_normal", activation = "relu", input_dim = 11))
model.add(Dropout(0.3))


# Adding the second hidden Layer

# In[ ]:


model.add(Dense(units=20, kernel_initializer="he_normal", activation = "relu"))
model.add(Dropout(0.4))


# Adding the third hidden layer

# In[ ]:


model.add(Dense(units=15, kernel_initializer="he_normal", activation = "relu"))
model.add(Dropout(0.2))


# Adding the Output Layer

# In[ ]:


model.add(Dense(units=1, kernel_initializer = "glorot_uniform", activation = "sigmoid"))


# In[ ]:


model.summary()


# Compiling the Artificial Neural Network Model

# In[ ]:


model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics =['accuracy'])


# Fitting the Neural Network Model to our Training Data

# In[ ]:


model_fit = model.fit(x_train, y_train, validation_split = 0.25, batch_size=10, epochs=100)


# List all the Data in history

# In[ ]:


print(model_fit.history.keys())


# In[ ]:


plt.plot(model_fit.history['accuracy'])
plt.plot(model_fit.history['val_accuracy'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend(['train','test'], loc = 'upper left')
plt.show()


# In[ ]:


plt.plot(model_fit.history['loss'])
plt.plot(model_fit.history['val_loss'])
plt.title("Model Accuracy")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend(['train','test'], loc = 'upper left')
plt.show()


# It's time to predict the test results now

# In[ ]:


y_pred = model.predict(x_test)


# In[ ]:


y_pred


# In[ ]:


y_pred = (y_pred>0.5)


# After predicting the output, our next task is to identify the performance of the model using different performance metrics

# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[ ]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[ ]:


acc_sc = accuracy_score(y_test, y_pred)
acc_sc

