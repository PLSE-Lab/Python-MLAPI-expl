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


# Importing Visualization Libraries

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Getting the Dataset

# In[ ]:


delhi_housing = pd.read_csv('/kaggle/input/delhi-house-price-prediction/MagicBricks.csv')


# In[ ]:


delhi_housing.head()


# In[ ]:


delhi_housing.info()


# In[ ]:


delhi_housing.describe()


# In[ ]:


delhi_housing.columns


# In[ ]:


sns.pairplot(delhi_housing)


# From paiplot we see that area,BHK,Bathroom has good correlation with price

# In[ ]:


sns.distplot(delhi_housing['Price'])


# In[ ]:


sns.heatmap(delhi_housing.corr(),annot=True)


# Heatmap shows result of pairplot in a better way

# Lets see how much of our Data is missing

# In[ ]:


sns.heatmap(delhi_housing.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# We can see there is lot of missing data in Per_Sqft and a little bit in Parking and seems like one in Bathroom.Lets handle it

# Per_Sqft is Price/Area as we have both these field therefore it is information duplication we can delete this column

# In[ ]:


delhi_housing.drop('Per_Sqft',axis=1,inplace=True)


# In[ ]:


sns.heatmap(delhi_housing.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


delhi_housing['Parking'].mean()


# we can see that Parking cloumn has an average value of 1.9 therefore we will be filling the missing parking values with 2(as parking can only be a whole no)

# In[ ]:


delhi_housing['Parking'].unique()


# In[ ]:


def average(parking):
    if pd.isnull(parking):
        return 2
    else:
        return parking


# In[ ]:


delhi_housing['Parking'] = delhi_housing['Parking'].apply(average)


# In[ ]:


delhi_housing['Parking'].unique()


# In[ ]:


sns.heatmap(delhi_housing.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# The no of missing value is very low we can now simply delete them

# In[ ]:


delhi_housing.dropna(inplace=True)


# In[ ]:


sns.heatmap(delhi_housing.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Adding Dummy variables for Categorical Columns

# In[ ]:


delhi_housing.info()


# In[ ]:


delhi_housing['Furnishing'].unique()


# In[ ]:


furnished = pd.get_dummies(delhi_housing['Furnishing'],drop_first=True)


# In[ ]:


delhi_housing['Status'].unique()


# In[ ]:


status = pd.get_dummies(delhi_housing['Status'],drop_first=True)


# In[ ]:


status


# In[ ]:


delhi_housing['Transaction'].unique()


# In[ ]:


transaction = pd.get_dummies(delhi_housing['Transaction'],drop_first=True)


# In[ ]:


delhi_housing['Type'].unique()


# In[ ]:


types = pd.get_dummies(delhi_housing['Type'],drop_first=True)


# In[ ]:


locality=pd.get_dummies(delhi_housing['Locality'],drop_first=True)


# In[ ]:


locality


# In[ ]:


delhi_housing.drop(['Furnishing','Status','Transaction','Type','Locality'],axis=1,inplace=True)


# In[ ]:


delhi_housing = pd.concat([delhi_housing,furnished,status,transaction,types,locality ],axis=1)


# In[ ]:


delhi_housing.head()


# In[ ]:


delhi_housing.columns


# seperating target columns from features

# In[ ]:


X = delhi_housing.loc[:, delhi_housing.columns != 'Price']
y = delhi_housing['Price']


# In[ ]:


X


# In[ ]:


y


# splitting the data into training and test set

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# applying LinerRegression

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lm = LinearRegression()


# In[ ]:


lm.fit(X_train,y_train)


# In[ ]:


print(lm.intercept_)


# In[ ]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[ ]:


predictions = lm.predict(X_test)


# In[ ]:


plt.scatter(y_test,predictions)


# Dont know why but the model performed very bad

# In[ ]:


sns.distplot((y_test-predictions),bins=50);


# In[ ]:


from sklearn import metrics


# In[ ]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# scaling the data to put into ANN

# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


scaler = MinMaxScaler()


# In[ ]:


X_train


# In[ ]:


X_train= scaler.fit_transform(X_train)


# In[ ]:


X_train


# In[ ]:


X_test = scaler.transform(X_test)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# training the model in ANN

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping


# In[ ]:


early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)


# In[ ]:


model = Sequential()

model.add(Dense(371,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(185,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(93,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(46,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')


# In[ ]:


model.fit(x=X_train,y=y_train.values,
          validation_data=(X_test,y_test.values),
          epochs=10000,callbacks=[early_stop])


# In[ ]:


losses = pd.DataFrame(model.history.history)


# In[ ]:


losses.plot()


# In[ ]:


from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score


# In[ ]:


predictions = model.predict(X_test)


# In[ ]:


mean_absolute_error(y_test,predictions)


# In[ ]:


np.sqrt(mean_squared_error(y_test,predictions))


# In[ ]:


explained_variance_score(y_test,predictions)


# got an explained variance of 73%

# In[ ]:


# Our predictions
plt.scatter(y_test,predictions)

# Perfect predictions
plt.plot(y_test,y_test,'r')


# atlest it performed better than LinearRegression

# Predicting the price of a single house

# In[ ]:


single_house = delhi_housing.drop('Price',axis=1).iloc[0]


# In[ ]:


single_house = scaler.transform(single_house.values.reshape(-1, 371))


# In[ ]:


delhi_housing['Price'][0]


# In[ ]:


model.predict(single_house)


# In[ ]:




