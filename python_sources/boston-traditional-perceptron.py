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


# Load Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Data Configuration
# 

# In[ ]:


boston = pd.read_csv('../input/boston-housing-dataset/HousingData.csv')
boston


# In[ ]:


boston.keys()


# **Data Set Characteristics:**  
# 
# :Number of Instances: 506 
# 
# :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.
# 
# :Attribute Information (in order)
#     - CRIM     per capita crime rate by town
#     - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
#     - INDUS    proportion of non-retail business acres per town
#     - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
#     - NOX      nitric oxides concentration (parts per 10 million)
#     - RM       average number of rooms per dwelling
#     - AGE      proportion of owner-occupied units built prior to 1940
#     - DIS      weighted distances to five Boston employment centres
#     - RAD      index of accessibility to radial highways
#     - TAX      full-value property-tax rate per \$10,000
#     - PTRATIO  pupil-teacher ratio by town
#     - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#     - LSTAT    \% lower status of the population
#     - MEDV     Median value of owner-occupied homes in $1000's

# In[ ]:


boston.info()


# In[ ]:


boston.isnull().sum()


# There are 20 null values in each [CRIM, ZN, INDUS, CHAS, AGE, LSTAT] columns.

# In[ ]:


boston[boston.CRIM.isnull()]


# In[ ]:


boston[boston.ZN.isnull()]


# And all null values are not located in same rows.

# In[ ]:


boston.describe()


# # EDA

# ## Comparing Crim Rate with Average Number of Rooms per Town

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

boston_CRIM = boston.CRIM
boston_RM = boston.RM

plt.scatter(boston_RM, boston_CRIM, marker = 'o')
plt.scatter(boston_RM.mean(), boston_CRIM.mean(), marker = '^')
plt.scatter(np.median(boston_RM), np.median(boston_CRIM), marker = 'v')

plt.title('Boston Crime-Rooms comparison Plot')
plt.ylabel('Crime Rate')
plt.xlabel('No. of Rooms')
plt.legend(['CR_relation','mean','median'])
plt.show()


# Theory I wanted to find in this plot
# - When the average number of Rooms goes up, the Crime rate goes down (downward direction graph)
# 
# Phenomena I found in this plot
# - There are some weird values which has extreamly high rate of crime. Except those values, the graph shows almost normal distribution and the mean and median values are located at center of the graph.
# - But still, the relation between average room numbers and crime rate is uncertain.

# ## Comparing Distances to Centeres with Pupil-Teacher Ratio per Town

# In[ ]:


boston_DIS = boston.DIS
boston_PTR = boston.PTRATIO

plt.scatter(boston_DIS, boston_PTR, marker = 'o', color = 'orange')
plt.scatter(boston_DIS.mean(), boston_PTR.mean(), marker = '^', color = 'black')
plt.scatter(np.median(boston_DIS), np.median(boston_PTR), marker = 'v', color = 'blue')

plt.title('Boston Distances to Centers - PTRATIO Comparison Plot')
plt.ylabel('Pupil-Teacher Ratio')
plt.xlabel('Distances to Centeres')
plt.legend(['DPT_relation', 'mean', 'median'])
plt.show()


# Theory I wanted to find in this plot
# - When the Distances to centeres goes up, the pupil-teacher ratio goes up (Upward direction graph)
# 
# Phenomena I found in this plot
# - All the values are scattered. Weak relations between Urbanism of area and Interest of Education of schools.

# # Preprocessing & Data Cleansing

# ## Box Plot of the values
# 
# see how the values are distributed and how's the outline.

# In[ ]:


X = boston.drop(columns = 'MEDV')
Y = boston.MEDV


# In[ ]:


X


# In[ ]:


# you could not make boxplot with dataframe. so, make it as numpy array.

X_np = X.to_numpy()


# In[ ]:


plt.boxplot(X_np)
plt.show()


# 1~4 and 7, 13th features are vacant in the graph because of the null values.\
# Let's fill the NA.

# ### Fill the NA
# 
# using dataframe.fillna

# In[ ]:


X.describe()


# In[ ]:


X_mean = X.fillna(X.mean())
X_mean.describe()


# In[ ]:


X_mean = X_mean.to_numpy()


# In[ ]:


plt.boxplot(X_mean)
plt.show()


# In[ ]:


boston.columns


# Now you can see those distributions of all columns.\
# But there are a huge gap in values between each columns.\
# So we need to scale the values!

# ## scaling the values
# 
# for make the values able to compare\
# also for the machine learning model albe to learn balanced.

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_scaler = scaler.fit_transform(X_mean)

plt.boxplot(X_scaler)
plt.show()


# [1st : CRIM,2nd : ZN,4th : CHAS,6th : RM,8th : DIS,11th : PTRATIO,12th : B,13th : LSTAT] columns have odds values.\
# So we need to make the values being in the extreme.\
# If the values are over the max value, make the values in the uppter extreme.\
# And if the values are under the lower extreme, make the values in the lower extreme.

# ![img](https://d2mvzyuse3lwjc.cloudfront.net/doc/en/UserGuide/images/Customizing_the_Box_Chart_Box_Tab_Controls/Customizing_the_Box_Chart_Box_Tab_Controls_4.png?v=2729)

# In[ ]:


X_scaler.shape


# ### Removing outlier

# In[ ]:


X_pd = pd.DataFrame(X_scaler, columns = X.columns)
X_pd


# In[ ]:


plt.boxplot(X_scaler)
plt.show()


# In[ ]:


X_pd[X_pd.CRIM > np.percentile(X_pd.CRIM,25)] = np.percentile(X_pd.CRIM,25)
X_pd[X_pd.CRIM < np.percentile(X_pd.CRIM,75)] = np.percentile(X_pd.CRIM,75)
X_pd2 = X_pd.to_numpy

plt.boxplot(X_pd2)
plt.show()
# np.percentile(X_pd.CRIM, 25), np.percentile(X_pd.CRIM, 75)


# In[ ]:


X_pd


# ## Data split

# In[ ]:


X = boston.drop(columns=['MEDV'])
Y = boston['MEDV']

from sklearn.model_selection import train_test_split

x_train_all, x_test, y_train_all, y_test = train_test_split(X,Y,test_size = 0.2)
x_train, x_val, y_train, y_val = train_test_split(x_train_all,y_train_all,test_size = 0.2)


# # Prediction

# ## Traditional - Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(x_train, y_train)
lr.score(x_test, y_test)


# In[ ]:


from sklearn import metrics

pred_lr = lr.predict(x_test)
metrics.r2_score(y_test, pred_lr)


# #### Linear Model - ElasticNet

# In[ ]:


from sklearn.linear_model import ElasticNet
en_9 = ElasticNet(alpha=0.9)
en_9.fit(x_train, y_train)
en_9.score(x_test, y_test)


# ## Perceptron

# In[ ]:


import tensorflow as tf
from tensorflow.keras import layers

regular = 0.1 # regularization amount

metrics_nm = ['accuracy','mean_squared_error']

model = tf.keras.Sequential()
model.add(layers.Input(shape=x_train.shape[1]))
model.add(layers.Dense(32, activation='relu',
         kernel_regularizer = tf.keras.regularizers.l2(regular),  # Dense Regularization
         activity_regularizer = tf.keras.regularizers.l2(regular)))  # Dense Regularization
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer='SGD', loss='mse', metrics=metrics_nm)

hist = model.fit(x_train, y_train, epochs=100, batch_size=128, validation_data=(x_val,y_val))


# **Veryfying by graph with weights & bias**

# In[ ]:


import matplotlib.pyplot as plt

weights, biases = model.layers[1].get_weights()
print(weights.shape, biases.shape)

plt.subplot(212)
plt.plot(weights,'x')
plt.plot(biases, 'o')
plt.title('L2 - 0.1')

plt.subplot(221)
plt.plot(hist.history['accuracy'],'^--',label='accuracy')
plt.plot(hist.history['val_accuracy'],'^--', label='v_accuracy')
plt.legend()
plt.title('L2 - 0.1')

plt.subplot(222)
plt.plot(hist.history['loss'],'x--',label='loss')
plt.plot(hist.history['val_loss'],'x--', label='val_loss')
plt.legend()
plt.title('L2 - 0.1')

plt.show()


# Model Fitting

# In[ ]:


model.fit(x_train, y_train, epochs=100, batch_size=16, validation_split=0.3)


# Before predict test data, need to scale

# In[ ]:


x_test = standardScaler.transform(x_test)


# In[ ]:


pred = model.predict(x_test)


# In[ ]:


model.evaluate(x_test, y_test, batch_size=16)


# ** Results **

# In[ ]:


from sklearn import metrics

metrics.r2_score(y_test, pred)


# In[ ]:




