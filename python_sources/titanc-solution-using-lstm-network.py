#!/usr/bin/env python
# coding: utf-8

# ## Titanc Solution using LSTM network##
# This is my first approach to solve the titanic Kaggle problem using LSTMs, of course improvements can be made.
# The main way the algorithm works is to use all the features as a time-series data. 
# I am open to comments and possible corrections on my code. 
# 

# ### Frameworks
# - pandas
# - numpy
# - seaborn 
# - keras
# - scikit-learn

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from keras.layers import Dense, LSTM, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import Adam


# In[ ]:


train = pd.read_csv('../input/train.csv', index_col = ["PassengerId"])
test = pd.read_csv('../input/test.csv', index_col = ["PassengerId"])
combination = [train,test]

train.head()


# The data at our disposal is numerical (age, Pclass, etc.), alphabetical (name, sex & embarked) and alphanumerical (tickets). Let's get some further insight on our data:

# In[ ]:


train.describe(), test.describe()


# From the data above we get information regarding the various means and standard deviation of each data, furthermore we can see where we have missing values. For example in the training data set there are 714 values for the passengers' ages, however, we know that the total number of passengers is 891. 
# This information will be useful later on. For now I will try to find which features will have a heavier influence on the Survival and delete those featurues that will have a lower influence on the output.

# In[ ]:


train[["Survived","Pclass"]].groupby("Pclass").mean()


# In[ ]:


train[["Survived", "Sex"]].groupby("Sex").mean()


# In[ ]:


train['groups']=pd.cut(train.Age,[0,10,20,30,40,50,60,70,80])
train.head()


# In[ ]:


train[["Survived", "groups"]].groupby("groups").mean()


# In the latter I have tried to divide the passengers by age group, by doing so we see how age has a big impact on the survival of the passenger, kids between 0-10 yrs have a higher survival rate. 

# ### Plotting ###
# To have a better view of the data I have decided to plot it. 

# In[ ]:


sns.barplot(train.groups, train.Survived)


# In[ ]:


sns.barplot(train.Sex, train.Survived)


# In[ ]:


sns.barplot(train.Pclass, train.Survived)


# In[ ]:


sns.barplot(train.Pclass, train.Survived, hue=train.Sex)


# In[ ]:


train.describe(include = ["O"]), test.describe(include = ["O"])


# For this version I have decided to delete the data regarding names, cabin and tickets. Cabin has many N/A values, names may not be directly related to the survival rate, however, their title could be relevant for future evaluations. In this case the name values are dropped, but in the future it could be interesting keeping the title of each passenger. 

# In[ ]:


train=train.drop(["Name", "Ticket","Cabin", "groups"], axis=1)
test=test.drop(["Name", "Ticket","Cabin"], axis=1)
train.head()


# ### Converting Data and filling missing data
# Sex are either male or females, hence we will convert this data to 1s and 0s respectively. Same thing can be applied to the embarked feature [0,1,2]. 

# In[ ]:


male_female = {"male":1,
              "female":0}

train["Sex"]=train["Sex"].map(male_female)
test["Sex"]=test["Sex"].map(male_female)
train.head()


# In[ ]:


embar = {"C":2,
        "S":1,
        "Q": 0}
train["Embarked"]=train["Embarked"].map(embar)
test["Embarked"]=test["Embarked"].map(embar)
train.head()


# In[ ]:



train["Age"]=train["Age"].fillna(value=np.mean(train["Age"]))
test["Age"]=test["Age"].fillna(value=np.mean(train["Age"]))
test["Fare"]=test["Fare"].fillna(value=np.mean(train["Fare"]))
train["Embarked"]=train["Embarked"].fillna(value=round(np.mean(train["Embarked"])))


# ## Model ##
# Extracting data

# In[ ]:


train_y = train["Survived"].iloc[:].values
train_x = train.drop(["Survived"], axis = 1).iloc[:,:].values

train_x = train_x.reshape(train_x.shape[0],-1,1)
train_x.shape


# LSTM parameters

# In[ ]:


batch_size = 11
epoch = 20
hidden_units = 256 


# LSTM architecture

# In[ ]:


model = Sequential()
model.add(LSTM(hidden_units, input_shape=train_x.shape[1:],batch_size=batch_size))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.compile(optimizer='Adam', loss = 'mean_squared_error',metrics = ['accuracy'] )
model.fit(train_x,train_y, batch_size=batch_size, epochs=epoch, verbose = 1)


# In[ ]:


out = pd.read_csv('../input/gender_submission.csv', index_col = ["PassengerId"])
y_test = out.iloc[:].values


# In[ ]:


test_x = test.iloc[:,:].values
test_x = test_x.reshape(test_x.shape[0],-1,1)
scores = model.evaluate(test_x, y_test, batch_size=batch_size)
predictions = model.predict(test_x, batch_size = batch_size)


# In[ ]:



print('LSTM test accuracy:', scores[1])


# 
