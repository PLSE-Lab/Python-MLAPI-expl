#!/usr/bin/env python
# coding: utf-8

# ## Apply NNet To Predict Stock Index and Trend ##
# 
# This script aims to analyze the Straits times Index (Singapore stock) index. 
# Following prediction needs to be made:
# 
# 1.  What is going to be trend tomorrow?
# 2. What will be the index tomorrow?
# 3.  If the user applied one dollar bet on each test data set and with correct prediction he gets 1.10 with wrong prediction he gets 0.90. On the test data set, will it lead to a loss or profit?

# In[7]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
stidf = pd.read_csv("../input/STI 2004-2013.csv")
stidf.head(5)


# ##Exploration of the Data
# Notice that the volume data is missing for many years. It will be wise to consider a model without volume column. 

# In[8]:


# Plot the trends for Open
subplots = stidf.plot(x="Date", subplots=True, figsize=(10,10))


# In[9]:


stidfnovol = stidf.drop("Volume", axis=1)


# In[10]:


stidfnovol.head()


# In[11]:


#Now we want to predict the tomorrow increment and trend. For this we would like to create some trend
#variables and then split our data in Train and test set.
#since it is a time series data we will need to handle it as timeseries

#Step 1: Let's create a timeseries out of dataframe
stidfnovol.index = pd.to_datetime(stidfnovol['Date'])
stidfnovol.dtypes


# **Let's Create some New trend variables**

# In[12]:


stidfnovol['tomoinc'] = stidfnovol['Close'].diff(periods=-1)
#stidfnovol = stidfnovol.drop('wkgain', axis=1)
stidfnovol['tomclose'] = stidfnovol['Close'] + stidfnovol['tomoinc']
stidfnovol['tomtrend'] = stidfnovol['tomoinc']/abs(stidfnovol['tomoinc'])
stidfnovol['tomoinc'] = (stidfnovol['tomoinc']*100)/stidfnovol['Close']
stidfnovol.head()


# ## Let's Plot the new dataframe with trend variables

# In[13]:


#Check if there are any NAN values
stidfnovol.dropna(axis=0, inplace=True)


# In[14]:


stidfnovol.isnull().sum()
stidfnovol.plot(x="Date", subplots=True, figsize=(15,15))


# ##Feature Engineering is complete. Let's Split the data in Train and Test##
# **Caution:** Since it's a timeseries data, we won't split in conventional random way. Let's split it first half in the trainset and mid to last as Testset

# In[15]:


#split the Data between dates 01-Jan-2000, 31-Dec-2011, 1-Jan-2012, 31-Dec-2020
import datetime
split_date = pd.datetime(2011,12,31)
stidfnovol['Date'] = stidfnovol['Date'].astype('datetime64[ns]')
df_train = stidfnovol.loc[stidfnovol['Date'] <= split_date]
df_test =  stidfnovol.loc[stidfnovol['Date'] > split_date]

#train_df = stidfnovol.between_time(datetime.time('2000-01-01'), datetime.time('2011-12-31'), include_start=True, include_end=True)
#train_df.head()


# In[16]:


df_train.iloc[:,1:8]
#df_train.drop('tomoinc', axis=1, inplace=True)


# In[17]:


df_train.iloc[:,6]


# In[18]:


df_test = df_test.drop('tomoinc', 1)
df_test.head()


# ##Let's Train the NNet Model and Fit our Train Data

# In[19]:


from keras.models import Sequential
from keras.layers import Dense


# ##Prepare the X and Y set in Train and Test##
# Relu function is also called a rectifier function. 

# ##Create X and Y##

# In[20]:


X_Train = df_train.iloc[:,1:7].reset_index()
X_Test = df_test.iloc[:,1:7].reset_index()
X_Test = X_Test.iloc[:,1:6]
X_Train = X_Train.iloc[:,1:6]
X_Train.head()


# In[21]:


X_Test.head()


# In[22]:


Y_Train = df_train.iloc[:,6].reset_index()
Y_Train = Y_Train.iloc[:,1]
Y_Test = df_test.iloc[:,6].reset_index()
Y_Test = Y_Test.iloc[:,1]
Y_Train.head()


# In[23]:


Y_Test.head()


# ##Create the Model##

# In[24]:


model = Sequential()
t2 = model.add(Dense(1, input_dim=5, activation='linear'))


# In[25]:


import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='rmsprop',
              loss='mse')
model


# In[26]:


model.fit(np.array(X_Train), 
          np.array(Y_Train), epochs=500, batch_size=10, verbose=0)


# ##Let's Predict the score for Test Set##

# In[27]:


scores = model.predict(np.array(X_Test), batch_size=10, verbose=0)


# In[28]:


#Let's Calculate the trend
predinc = scores - Y_Test[1]
predtrend = predinc/abs(predinc)
predtrend
df_test.iloc[:,7]


# In[29]:


#Let's plot the confusion Matrix
from sklearn.metrics import confusion_matrix
conf = confusion_matrix(df_test.iloc[:,7], predtrend)
conf


# In[30]:


conf[1,0]


# ##Here is the Calculation if the person is going to make Money or not :-)##

# In[31]:


LossOrProfit = conf[0,0]*1.1 + conf[1,1]*.1 - conf[0,1]*.9 - conf[1,0]*.9
LossOrProfit

