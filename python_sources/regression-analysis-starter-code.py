#!/usr/bin/env python
# coding: utf-8

# **IMPORT LIBRARIES**

# In[ ]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# **Loading the data**

# In[ ]:


data = pd.read_csv("/kaggle/input/autompg-dataset/auto-mpg.csv")
data.head()


# In[ ]:


data.shape


# Dataset have 398 rows and 9 columns

# In[ ]:


data.columns


# Lets look at data types:-
# 
# mpg: continuous
# cylinders: multi-valued discrete
# displacement: continuous
# horsepower: continuous
# weight: continuous
# acceleration: continuous
# model year: multi-valued discrete
# origin: multi-valued discrete
# car name: string (unique for each instance)

# **Lets look at if there's any null values in the data**

# In[ ]:


data.isnull().sum()


# In[ ]:


data.index


# In[ ]:


data.tail()


# **Lets look more into the data**

# In[ ]:


data.describe()


# In[ ]:


data.info()


# Since there are no null values in the dataset, the column 'horsepower' contains '?', so we replace it with horsepower 100

# In[ ]:


data['horsepower'] = data['horsepower'].replace('?','100')


# In[ ]:


data['horsepower'].value_counts()


# ****DATA PREPROCESSING****

# In[ ]:


data.head()


# ****Lets have a look at mpg****

# In[ ]:


#mpg as factors
print('Highest mpg is',data.mpg.max(),'millions per gallon')
print('Lowest mpg is',data.mpg.min(),'millions per gallon')


# In[ ]:


f,ax = plt.subplots(1,2,figsize=(12,6))
sns.boxplot(data.mpg,ax=ax[0])
sns.distplot(data.mpg,ax=ax[1])


# In[ ]:


print("Skewness: ",data['mpg'].skew())
print("Kurtosis: ",data['mpg'].kurtosis())


# ****CORRELATION****
# 

# In[ ]:


corr = data.corr()
corr


# In[ ]:


data.corr()['mpg'].sort_values()


# In[ ]:


plt.figure(figsize=(12,5))
sns.heatmap(corr,annot = True,cmap = 'Accent',linewidths = 0.2 )


# In[ ]:


## multivariate analysis
sns.boxplot(y='mpg',x='cylinders',data=data)
plt.show()


# **Lets check our categorical column (car name)**

# In[ ]:


data['car name'].describe()


# In[ ]:


data['car name'].value_counts()


# In[ ]:


data['car name'].unique()


# while looking into the car names there are 305 different cars are in the data so we sort them as the company names 

# In[ ]:


data['car name'] = data['car name'].str.split(' ').str.get(0)
data['car name'].value_counts()


# some company needs replace in their name,

# In[ ]:


data['car name'] = data['car name'].replace(['chevroelt','chevy'],'chevrolet')
data['car name'] = data['car name'].replace(['vokswagen','vw'],'volkswagen')
data['car name'] = data['car name'].replace('maxda','mazda')
data['car name'] = data['car name'].replace('toyouta','toyota')
data['car name'] = data['car name'].replace('mercedes','mercedes-benz')
data['car name'] = data['car name'].replace('nissan','datsun')
data['car name'] = data['car name'].replace('capri','ford')
data['car name'].value_counts()


# In[ ]:


plt.figure(figsize=(15,8))
sns.countplot(data['car name'])
plt.xticks(rotation = 90)


# In[ ]:


sns.scatterplot(x='cylinders',y='displacement',hue = 'mpg',data=data,cmap = 'rainbow')


# ****MACHINE LEARNING****

# In[ ]:


x = data.iloc[:,1:].values
y = data.iloc[:,0].values
x


# In[ ]:


#Encoding categorical data
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
x[:,7] = lb.fit_transform(x[:,7])

from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder(categorical_features = [7])
x = onehot.fit_transform(x).toarray()


# In[ ]:


#Splitting into training and test data
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.2,random_state = 0)


# In[ ]:


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)


# ****REGRESSION****
# 1. Multiple linear Regression
# 2. Decision Tree Regression
# 3. Random Forest Regression
# 4. Support Vector Regression

# **1. Multiple linear Regression**

# In[ ]:


# multiple linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(xtrain,ytrain)
ypred = lr.predict(xtest)

lr.score(xtrain,ytrain)

from sklearn.metrics import r2_score
print("Accuracy of the linear model is:",round(r2_score(ytest,ypred)*100,2),'%')


# **2. Decision Tree Regressor**

# In[ ]:


#Decision tree Regression
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state = 0)
dtr.fit(xtrain,ytrain)

ypred_dtr = dtr.predict(xtest) 
print('Accuracy of the decision tree model is:',round(r2_score(ytest,ypred_dtr)*100,2),'%')


# **3. Random Forest Regressor**
# 

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 200,random_state = 0)
rfr.fit(xtrain,ytrain)

ypred_rfr = rfr.predict(xtest)
print('Accuracy of the random forest model:',round(r2_score(ytest,ypred_rfr)*100,2),'%')


# ****4. Support Vector Regression****

# In[ ]:


#Support Vector Regression
from sklearn.svm import SVR
svr = SVR(kernel = 'rbf',gamma = 'scale')
svr.fit(xtrain,ytrain)

ypred_svr = svr.predict(xtest)
print('Accuracy of the SVR model :',round(r2_score(ytest,ypred_svr)*100,2),'%')


# **So amoung the 4 models random forest regressor is the best model with an accuracy of 89.96%**

# Did you find this Notebook useful?
# Show your appreciation with an upvote
