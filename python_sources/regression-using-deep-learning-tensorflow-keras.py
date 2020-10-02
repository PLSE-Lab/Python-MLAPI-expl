#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.simplefilter('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Any results you write to the current directory are saved as output.


# In[2]:


data = pd.read_csv('../input/kc_house_data.csv')


# In[3]:


data.head()


# In[4]:


data.describe()


# In[5]:


data.info()


# In[6]:


data.head()


# In[7]:


data.isnull().values.any()


# In[8]:


import datetime
def get_year(date):
    date = str(date)
    year = date[0:4]
    month = date[4:6]
    day = date[6:8]
    date = year+'-'+month+'-'+day
    date = datetime.datetime.strptime(date, '%Y-%m-%d')    
    return date


# In[9]:


data['date'] = data['date'].apply(get_year)


# In[10]:


data.head()


# A point to notice is that square feet living = sq. feet above + sq. feet basement. What we can rather do is have a categorical column for if there is a basement of not and delete the sq. feet above and sq. feet basement columns altogether. Afterall what matters is the total living area and if there is a basement there or not. The area of the basement would not have a major affect of its own as such as it is already included in the total living area

# In[11]:


data['is_basement'] = data['sqft_basement'].apply(lambda x: 1 if x != 0 else 0)


# In[12]:


#updating living room area and lot area based on the new values of 2015
data = data.drop(['sqft_living','sqft_lot','sqft_basement','sqft_above'],axis=1)


# In[13]:


data.head()


# In[14]:


def update_yr_built(cols):
    yr_built = cols[0]
    yr_renovated = cols[1]
    if yr_renovated != 0:
        yr_built = yr_renovated
        
    return yr_built
data['yr_built'] = data[['yr_built','yr_renovated']].apply(update_yr_built,axis=1)


# In[15]:


data.head()


# In[16]:


data = data.drop('yr_renovated',axis=1)


# In[17]:


data.head()


# In[18]:


data = data.drop(['id','view','lat','long'],axis=1)
data.head()


# In[19]:


sns.pairplot(data.drop(['date','condition','yr_built','zipcode','waterfront','is_basement'],axis=1))


# In[20]:


data.groupby('waterfront').mean()


# That's weird - An avg house having a waterfront is larger in size but way more cheaper. Maybe houses having a waterfront are in the outskirts where the prices are very low.

# In[21]:


#percentage of households having a waterfront
(data['waterfront'].sum()/len(data))*100


# Not even 1% of the households that we have, have a waterfront.

# In[22]:


data.groupby('is_basement').mean()


# Having a basement is a plus point. It increases the prices! (Area of living is almost the same)

# In[23]:


#percentage of households having a basement
(data['is_basement'].sum()/len(data))*100


# In[24]:


data.head()


# In[25]:


plt.figure(figsize=(10,6))
sns.lineplot(x='date',y='price',data=data,ci=False)


# The prices have remained fairly consistent throughout both the years.

# In[26]:


data = data.drop('date',axis=1)


# Lets predict the prices 

# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


X = data.drop('price',axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[29]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[30]:


norm_X_train = scaler.fit_transform(X_train)


# In[31]:


norm_X_test = scaler.fit_transform(X_test)


# In[32]:


norm_X_train = pd.DataFrame(data=norm_X_train,columns=X_train.columns)


# In[33]:


norm_X_test = pd.DataFrame(data=norm_X_test,columns=X_test.columns)


# In[34]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers


# In[35]:


def build_model():
    model = keras.Sequential([
        layers.Dense(60, activation=tf.nn.relu, input_shape=[len(norm_X_train.keys())]),
        layers.Dropout(0.5),
        layers.Dense(60, activation=tf.nn.relu,kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)),
        layers.Dropout(0.5),
        layers.Dense(60, activation=tf.nn.relu,kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)),
        layers.Dense(1)
          ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    
    model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


# In[36]:


model = build_model()


# In[37]:


model.summary()


# In[38]:


EPOCHS = 1500


# In[39]:


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
  
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
    
    plt.legend()
  
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
    
    plt.legend()
    plt.show()

#plot_history(history)


# In[40]:


#hist = pd.DataFrame(history.history)
#hist['epoch'] = history.epoch
#hist.tail()


# In[41]:


model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(norm_X_train, y_train, epochs=EPOCHS,batch_size=100,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop])


# In[42]:


plot_history(history)


# In[43]:


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


# In[ ]:





# In[44]:


loss, mae, mse = model.evaluate(norm_X_train, y_train, verbose=0)

print("Training set Mean Abs Error: {:5.2f}".format(mae))


# In[45]:


data['price'].mean()


# In[46]:


test_predictions = model.predict(norm_X_test).flatten()
plt.figure(figsize=(10,6))
plt.scatter(y_test, test_predictions)
plt.xlabel('True prices')
plt.ylabel('Predicted prices')
plt.xlim(0,2000000)
plt.ylim(0,2000000)


# In[47]:


loss, mae, mse = model.evaluate(norm_X_test, y_test, verbose=0)

print("Testing set Mean Abs Error: {:5.2f}".format(mae))


# In[48]:


#using linear regression - 
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(norm_X_train,y_train)
predictions = lm.predict(norm_X_test)
plt.figure(figsize=(10,6))
ax = plt.scatter(y_test,predictions)
plt.xlim(0,2000000)


# In[49]:


from sklearn import metrics
metrics.mean_absolute_error(y_test,predictions)


# In[ ]:




