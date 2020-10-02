#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy


# In[ ]:


#Importing the dataset
dataset = pd.read_csv('CarPrice_Assignment.csv')


# In[ ]:


dataset


# In[ ]:


# Remove Car_ID according to businesss logic
dataset = dataset.iloc[:,1:]
dataset


# In[ ]:


# check for null value
dataset.isnull().sum()


# In[ ]:


# check for nan value
dataset.isna().sum()


# In[ ]:


dataset.describe()


# In[ ]:


# wheel base
plt.scatter(dataset.index, dataset.wheelbase)
plt.show()


# In[ ]:


sns.boxplot(x=dataset.wheelbase)


# In[ ]:


#replace the outliers by mean
li = dataset[dataset.wheelbase >=115].index
dataset['wheelbase'][li] = float(dataset.drop(li)['wheelbase'].mean())

# Check the result
dataset[dataset.wheelbase>=115]


# In[ ]:


# Car length
sns.boxplot(x=dataset.carlength)


# In[ ]:


plt.scatter(dataset.carlength, dataset.index)
plt.show()


# In[ ]:


#replace the outliers by mean
li = dataset[dataset.carlength <142].index
dataset['carlength'][li] = float(dataset.drop(li)['carlength'].mean())

# Check the result
dataset[dataset.carlength<142]


# In[ ]:


# Car width
sns.boxplot(x=dataset.carwidth)


# In[ ]:


plt.scatter(dataset.carwidth,dataset.index)
plt.show()


# In[ ]:


#replace the outliers by mean
li = dataset[dataset.carwidth >71].index
dataset['carwidth'][li] = float(dataset.drop(li)['carwidth'].mean())

# Check the result
dataset[dataset.carwidth>71]


# In[ ]:


#Car height
sns.boxplot(x=dataset.carheight)


# In[ ]:


plt.scatter(dataset.carheight,dataset.index)


# In[ ]:


# Curb weight
sns.boxplot(x=dataset.curbweight)


# In[ ]:


plt.scatter(dataset.curbweight,dataset.index)


# In[ ]:


# Engine size
plt.scatter(dataset.enginesize,dataset.index)


# In[ ]:


sns.boxplot(x=dataset.enginesize)


# In[ ]:


#replace the outliers by mean
li = dataset[dataset.enginesize >200].index
dataset['enginesize'][li] = float(dataset.drop(li)['enginesize'].mean())

# Check the result
dataset[dataset.enginesize>200]


# In[ ]:


# bore ratio
sns.boxplot(x=dataset.boreratio)


# In[ ]:


plt.scatter(dataset.boreratio,dataset.index)


# In[ ]:


# stroke
sns.boxplot(x=dataset.stroke)


# In[ ]:


plt.scatter(dataset.stroke,dataset.index)


# In[ ]:


#replace the outliers by mean
li = dataset[dataset.stroke>3.8].index
li2 = dataset[dataset.stroke<2.5].index
li = np.concatenate((li,li2))


# In[ ]:


dataset['stroke'][li] = float(dataset.drop(li)['stroke'].mean())

# Check the result
dataset[dataset.stroke>3.8]


# In[ ]:


# Compresion ratio
sns.boxplot(x=dataset.compressionratio)


# In[ ]:


dataset[dataset.compressionratio>11]['compressionratio']


# In[ ]:


dataset[dataset.compressionratio<7.5]['compressionratio']


# In[ ]:


#replace the outliers by mean
li = dataset[dataset.compressionratio>11].index
li2 = dataset[dataset.compressionratio<7.5].index
li = np.concatenate((li,li2))
li


# In[ ]:


dataset['compressionratio'][li] = float(dataset.drop(li)['compressionratio'].mean())

# Check the result
dataset[dataset.compressionratio>11]


# In[ ]:


# Horsepower
sns.boxplot(x=dataset.horsepower)


# In[ ]:


#replace the outliers by mean
li = dataset[dataset.horsepower >190].index
dataset['horsepower'][li] = float(dataset.drop(li)['horsepower'].mean())

# Check the result
dataset[dataset.horsepower>190]


# In[ ]:


# RPM
sns.boxplot(x=dataset.peakrpm)


# In[ ]:


#replace the outliers by mean
li = dataset[dataset.peakrpm >6500].index
dataset['peakrpm'][li] = float(dataset.drop(li)['peakrpm'].mean())

# Check the result
dataset[dataset.peakrpm>6500]


# In[ ]:


# city mileage
sns.boxplot(x=dataset.citympg)


# In[ ]:


#replace the outliers by mean
li = dataset[dataset.citympg >45].index
dataset['citympg'][li] = float(dataset.drop(li)['citympg'].mean())


# In[ ]:


dataset[dataset.citympg>45]['citympg']


# In[ ]:


# Highway mileage
sns.boxplot(x=dataset.highwaympg)


# In[ ]:


#replace the outliers by mean
li = dataset[dataset.highwaympg >=50].index
dataset['highwaympg'][li] = float(dataset.drop(li)['citympg'].mean())

#check the result
dataset[dataset.highwaympg >=50]


# In[ ]:


# changing the datatype of symboling as it is categorical variable as per dictionary file

dataset['symboling'] = dataset['symboling'].astype(str)
dataset['doornumber'] = dataset['doornumber'].astype(str)


# In[ ]:


# Unique Car company
dataset['CarName'] = dataset['CarName'].str.split(' ',expand=True)
dataset['CarName'] = dataset['CarName'].replace({'maxda': 'mazda', 'nissan': 'Nissan', 'porcshce': 'porsche', 'toyouta': 'toyota', 
                            'vokswagen': 'volkswagen', 'vw': 'volkswagen'})
dataset.CarName


# In[ ]:


dataset


# In[ ]:


# split
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]


# In[ ]:


print(X.shape)
print(y.shape)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
for i in X.columns:
    if isinstance(X[i][0], str):
            X[i] = encoder.fit_transform(X[i])


# In[ ]:


# VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

X = add_constant(X)

pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns).sort_values()


# In[ ]:


# Symboling
import scipy.stats as stats
stats.f_oneway(X.symboling,y)


# In[ ]:


# Door number
stats.f_oneway(X.doornumber,y)


# In[ ]:


# Car company
stats.f_oneway(X.CarName,y)


# In[ ]:


# Fuel type
stats.f_oneway(X.fueltype,y)


# In[ ]:


# Aspiration
stats.f_oneway(X.aspiration,y)


# In[ ]:


# Car body
stats.f_oneway(X.carbody,y)


# In[ ]:


# Drive wheel
stats.f_oneway(X.drivewheel,y)


# In[ ]:


# Engine location
stats.f_oneway(X.enginelocation,y)


# In[ ]:


# Engine type
stats.f_oneway(X.enginetype,y)


# In[ ]:


# Cylinder number
stats.f_oneway(X.cylindernumber,y)


# In[ ]:


# Fuel system
stats.f_oneway(X.fuelsystem,y)


# In[ ]:


X2 = copy.copy(X)


# In[ ]:


X2.columns


# In[ ]:


X2.drop((['symboling', 'CarName', 'fueltype', 'aspiration',
       'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem']),axis=1,inplace=True)


# In[ ]:


X2.drop(('doornumber'),axis=1,inplace=True)


# In[ ]:


# Cofficient corelation
correlation = pd.Series([np.corrcoef(X2[i],y)[0,1] 
                         for i in X2.columns], index=X2.columns)
correlation


# In[ ]:


#Removing less significant columns (value >=0.5)
X.drop((['const','carlength','carwidth','curbweight','enginesize','boreratio']),axis=1,inplace=True)
X.columns


# In[ ]:


# Train test split
from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=0.2,random_state=0)


# In[ ]:


# Multiple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


# In[ ]:


# Training the model
regressor.fit(train_X,train_y)


# In[ ]:


# Prediction
y_pred = regressor.predict(test_X)
y_pred


# In[ ]:


#Training accuracy
regressor.score(train_X,train_y)*100


# In[ ]:


#Testing accuracy
regressor.score(test_X,test_y)*100


# In[ ]:




