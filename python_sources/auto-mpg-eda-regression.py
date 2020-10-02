#!/usr/bin/env python
# coding: utf-8

# **Importing the Libraries**

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


# **Loading the dataset**

# In[ ]:


data = pd.read_csv("../input/auto-mpg.csv")
data.head()


# In[ ]:


# Making the copy of the dataframe
df = data.copy


# Dropping the categorical feature from the dataframe for further analysis

# In[ ]:


data.drop(['car name'],axis=1,inplace=True)
data.head()


# **Summary of the dataset**

# In[ ]:


data.describe()


# **Data Preprocessing**

# 1. Checking for null values in the dataset

# In[ ]:


data.isnull().sum()


# In[ ]:


data['horsepower'].unique()


# * There are** no null values** in our dataset.
# But the ** horsepower ** feature in our dataframe contains '?' which need to be removed from the dataframe , so we will **drop the rows **in the dataframe where horsepower is equal to '?'.

# In[ ]:


data = data[data.horsepower != '?']


# In[ ]:


# Checking for null values after dropping the rows
'?' in data


# In[ ]:


data.shape


# After dropping the rows containing horsepower as '?', now we are left with 392 rows.

# Here , I am checking the correlation of all the features of the dataset w.r.t miles-per-gallon i.e.** 'mpg'** and arranging the values in ascending order.

# In[ ]:


data.corr()['mpg'].sort_values()


# **Heatmap of correlation matrix**

# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,linewidth=0.5,center=0,cmap='rainbow')
plt.show()


# **Univariate Analysis**

# In[ ]:


sns.countplot(data.cylinders,data=data,palette = "rainbow")
plt.show()


# From the above above plot we can visualize that there are ** maximum number of 4 cylinder** vehicles.
# Around **98% of the vehicles are either of 4, 6, 8 cylinders** and only small percent of vehicles are either  of 3 and 5 cylinders.

# In[ ]:


sns.countplot(data['model year'],palette = "rainbow")
plt.show()


# **Maximum number of** vehicles are of the **year 1973** and **minumum number** of the **year 1974**.

# In[ ]:


sns.countplot(data.origin,palette = "rainbow")
plt.show()


# Most of the vehicles are from region 1 as compared to the other two regions.

# In[ ]:


data['horsepower'] = pd.to_numeric(data['horsepower'])
sns.distplot(data['horsepower'])
plt.show()


# 1. Horsepower rates the engine performance of cars
# 2. From the above plot we can see the distribution of the horsepower of the vehicles.
# 3. We can visualize that most of the vehicles have around** 75-110 horsepower** and only few vehicles have horsepoer above 200.

# **Engine displacement** is the swept volume of all the pistons inside the cylinders of a reciprocating engine in a single movement from top dead centre (TDC) to bottom dead centre (BDC).

# In[ ]:


sns.distplot(data.displacement,rug=False)
plt.show()


# In[ ]:


## multivariate analysis
sns.boxplot(y='mpg',x='cylinders',data=data,palette = "rainbow")
plt.show()


# We can easily visualize that the mileage per gallon (mpg) of 4 cylinder vehicles is maximum and we also saw that most of the vehicles are 4 cylinder.
# - From the above result we can carry out the inference that for most of the people** mileage(mpg) **is one of the major factor while buying a vehicle.

# In[ ]:


sns.boxplot(y='mpg',x='model year',data=data,palette = "rainbow")
plt.show()


# With every year and with the newer models of the vehicles mileage per gallon (mpg) also increases.

# In[ ]:


plot = sns.lmplot('horsepower','mpg',data=data,hue='origin',palette = "rainbow")
plt.show()


# In[ ]:


plot = sns.lmplot('acceleration','mpg',data=data,hue='origin',palette = "rainbow")
plot.set(ylim = (0,50))
plt.show()


# In[ ]:


plot = sns.lmplot('weight','mpg',data=data,hue='origin',palette = "rainbow")
plt.show()


# In[ ]:


plot = sns.lmplot('displacement','mpg',data=data,hue='origin',palette = "rainbow")
plot.set(ylim = (0,50))
plt.show()


# ** Modelling**

# In[ ]:


X = data.iloc[:,1:].values
Y = data.iloc[:,0].values


# Splitting the dataset into training and test set.

# **1. Multivariate Regression**

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)

regressor = LinearRegression()
regressor.fit(X_train,Y_train)


# In[ ]:


Y_pred = regressor.predict(X_test)
print(regressor.score(X_test,Y_test))


# **Polynomial regression**

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)

X_train,X_test,Y_train,Y_test = train_test_split(X_poly,Y,test_size=0.30)

lin_reg = LinearRegression()
lin_reg  = lin_reg.fit(X_train,Y_train)

print(lin_reg.score(X_test,Y_test))


# **Thank You**
