#!/usr/bin/env python
# coding: utf-8

# ## This project is mainly concerned about Weather Prediction of an year is El-Nina or La-Nina year using LSTM Model.
# Take aways from this project is
# * This project is helpful to build the skills in Data Analysis for new programmers.
# * Using Time Series for Weather Prediction.
# * Data Preprocessing.
# * Modelling LSTM.

# In[1]:


# Import all required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import keras as kr
import sklearn
import math
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import itertools
import warnings
warnings.filterwarnings('ignore')


# In[3]:


data = pd.read_csv('../input/El-Nino.csv', sep = '\t')


# In[4]:


data.head()


# ## Data Preprocessing
# ### Replace all column names by overwritting on it

# In[5]:


cols = ['Year','Janauary','February','March','April','May','June','July','August','September','October','November','December']


# In[6]:


data.columns = cols


# In[7]:


data.head(10)


# ### Set Index as Year, it is because by seeing the data one can understand is the problem is related to Time Series Forecasting

# In[9]:


data.set_index('Year', inplace = True)
data.head()


# ### Do transpose to know, how many years are present

# In[10]:


data1 = data.transpose()
data1


# ### Generate the date_range series. It is because converting whole 12 attributes into single series of each month by considering length of data1.columns multiplied by 12.

# In[11]:


dates = pd.date_range(start = '1950-01', freq = 'MS', periods = len(data1.columns)*12)
dates


# ### Convert the dataframe into matrix 

# In[15]:


data_np = data1.transpose().as_matrix()
shape = data_np.shape
data_np


# ### Let's convert the matrix size of 68 x 12 into column vector 

# In[16]:


data_np = data_np.reshape((shape[0] * shape[1], 1))
data_np.shape


# ### Convert the data_np into dataframe
# * Here we are merging two series data i.e data_np and dates series into dataframe.
# * As this dataset belongs to timeseries concept, we apply dates series as index to our dataframe.

# In[17]:


df = pd.DataFrame({'Mean' : data_np[:,0]})
df.set_index(dates, inplace = True)
df.head()


# ### Now Let's plot how our data looks like

# In[18]:


plt.figure(figsize = (15,5))
plt.plot(df.index, df['Mean'])
plt.title('Yearly vs Monthly Mean')
plt.xlabel('Year')
plt.ylabel('Mean across Month')


# In[19]:


dataset = df.values
dataset.shape


# ### Here we are splitting the data into train and test set

# In[20]:


train = dataset[0:696,:]
test = dataset[696:,:]


# In[21]:


print("Original data shape:",dataset.shape)
print("Train shape:",train.shape)
print("Test shape:",test.shape)


# ### LSTM Modelling

# In[22]:


# Converting the data into MinMax Scaler because to avoid any outliers present in our dataset
scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data.shape


# ### As we know we use LSTM model to our data then we follow Imporvements over RNN principle
# * To see more inbrief Click [here](https://www.analyticsvidhya.com/blog/2017/12/fundamentals-of-deep-learning-introduction-to-lstm/)

# In[23]:


x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)


# In[24]:


#x_train shape
x_train.shape


# In[25]:


#y_train shape
y_train.shape


# In[27]:


x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
x_train.shape


# In[28]:


# Creating and fitting the model
model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1],1)))
model.add(LSTM(units = 50))
model.add(Dense(1))


# In[29]:


model.compile(loss = 'mean_squared_error', optimizer = 'adam')
model.fit(x_train, y_train, epochs=10, batch_size = 1, verbose = 2)


# In[30]:


# Now Let's perform same operations that are done on train set
inputs = df[len(df) - len(test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)


# In[31]:


X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)


# In[32]:


X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
Mean = model.predict(X_test)
Mean1 = scaler.inverse_transform(Mean)


# In[33]:


# Check for the RMS error between test set and Mean1 predicted values
rms=np.sqrt(np.mean(np.power((test-Mean1),2)))
rms


# In[34]:


#plotting the train, test and forecast data
train = df[:696]
test = df[696:]
test['Predictions'] = Mean1

plt.figure(figsize=(15,5))
plt.plot(train['Mean'])
plt.plot(test['Mean'], color = 'black')
plt.plot(test['Predictions'], color = 'orange')
plt.xlabel('Years')
plt.ylabel('Mean')
plt.title('Forecasting on Actual data')


# In[35]:


# Here we are taking steps as 2, means we have taken test size as 120 that is step=1.  
#steps=2 means taking 120 test values and 120 future values i.e next 10 year values from test data
trainpred = model.predict(X_test,steps=2)


# In[36]:


trainpred.shape


# In[37]:


pred = scaler.inverse_transform(trainpred)


# In[38]:


# Total predicted values are 240, but now I'm printing only first 24 values
pred[0:24] 


# In[39]:


test.head()


# In[40]:


# Now printing the test Accuracy
testScore = math.sqrt(mean_squared_error(test['Mean'], trainpred[:120,0]))*100
print('Accuracy Score: %.2f' % (testScore))


# In[41]:


# Now consider which year we want to predict the value
# Here enter the year which should be greater than 2017 i.e above test set values
step_yr = 2017
yr = int(input('Enter the Year to Predict:'))
c = yr - step_yr
e = c-1
b = pred[120+(e*12) : 120+(e*12)+12].mean(axis=0)


# ### Inorder to check how these values are adjusted, you can check from [here](https://ggweather.com/enso/oni.htm)
# 

# In[48]:


print(b)
if b >= 0.5 and b <= 0.9:
    print(yr, 'is Weak El-Nino')
elif b >= 1.0 and b <= 1.4:
    print('It is Moderate El-Nino')
elif b >= 1.5 and b <= 1.9:
    print(yr, 'is Strong El-Nino')
elif b >= 2:
    print(yr, 'is Very Strong El-Nino')
elif b <=-0.5 and b >= -0.9:
    print(yr, 'is Weak La-Nina')
elif b <= -1 and b >= -1.4:
    print(yr, 'is Moderate La-Nina')
elif b <= -1.5:
    print(yr, 'is Strong La-Nina')
else:
    print(yr, 'is a Moderate Year')


# In[43]:


# Now plot the graph of future predicted values for that generate a date range series upto 2027
dates1 = pd.date_range(start = '2008-01', freq = 'MS', end = '2027-12')
dates1


# In[44]:


new_df = pd.DataFrame({'Predicted_values':pred[:,0]})


# In[45]:


new_df.set_index(dates1, inplace = True)


# In[46]:


new_df.head()


# In[47]:


# Now plot the graph to see how over train, test and future values are predicted
plt.figure(figsize=(15,5))
plt.plot(train['Mean'])
plt.plot(test['Mean'], color = 'black')
plt.plot(test['Predictions'], color = 'orange')
plt.plot(new_df['Predicted_values'][120:], color = 'red')
plt.xlabel('Years')
plt.ylabel('Mean')
plt.legend(loc = True)
plt.title('Forecasting on Actual data')


# In[ ]:




