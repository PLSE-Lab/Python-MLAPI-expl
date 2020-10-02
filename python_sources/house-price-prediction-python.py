#!/usr/bin/env python
# coding: utf-8

# House Price Prediction using Linear Regression
# ----------------------------------------------

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data= pd.read_csv("../input/kc_house_data.csv")
data.head()


# In[ ]:


data.describe(include=[np.number])


# In[ ]:


data.isnull().sum()  #Data not having any NaNs


# In[ ]:


names=['price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','zipcode','lat','long']
df=data[names]
correlations= df.corr()
fig=plt.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(correlations,vmin=-1,vmax=1)
fig.colorbar(cax)
ticks=np.arange(0,15,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()


# Top 5 features:
# 
#  1. bedrooms
#  2. bathrooms
#  3. sqft_living
#  4. sqft_above
#  5. grade

# Now Let's convert nominal and ordinal features into category.
# 

# In[ ]:


data['waterfront'] = data['waterfront'].astype('category',ordered=True)
data['view'] = data['view'].astype('category',ordered=True)
data['condition'] = data['condition'].astype('category',ordered=True)
data['grade'] = data['grade'].astype('category',ordered=False)
data['zipcode'] = data['zipcode'].astype('category',ordered=False)


# In[ ]:


data.dtypes


# ## Exploratory Analysis ##

# In[ ]:


#sns.set_style()
sns.regplot(x='sqft_living',y='price',data=data)


# In[ ]:


sns.regplot(x='sqft_basement',y='price',data=data)


# In[ ]:


sns.regplot(x='sqft_above',y='price',data=data)


# In[ ]:


sns.stripplot(x='bedrooms', y='price',data=data)


# The graph indicates that the relation of price with bedrooms. Price goes up with number of bedrooms upto 6 and then comes down as no of bedrooms increase.

# In[ ]:


sns.stripplot(x='bathrooms', y='price',data=data, size=5)


# In[ ]:


sns.stripplot(x='grade', y='price',data=data, size=5)


# Price of the house is dependent on grade of the house, higher the grade higher the price.

# We can remove outliers from data like house with bedrooms > 9 and bathrooms>7.

# In[ ]:


data=data[data['bedrooms'] < 10]


# In[ ]:


data=data[data['bathrooms']<8]


# In[ ]:


data.head()


# Building a model with top 5 parameters
# ----------------

# In[ ]:


c=['bedrooms','bathrooms','sqft_living','sqft_above','grade']
df=data[c]


# In[ ]:


df=pd.get_dummies(df,columns=['grade'], drop_first=True)


# In[ ]:


y=data['price']


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(df,y,train_size=0.8,random_state=42)


# In[ ]:


x_train.head()


# In[ ]:


reg=LinearRegression()


# In[ ]:


reg.fit(x_train,y_train)


# In[ ]:


print('Coefficients: \n', reg.coef_)
#Coefficients: 
# [ -2.29668616e+04  -2.58269301e+03   2.31278874e+02  -9.49870688e+01
# -5.30622026e+05  -4.81879701e+05  -4.82543663e+05  -4.51474787e+05
# -4.19169329e+05  -3.40826437e+05  -1.99835359e+05   7.81207530e+03
#  2.84775801e+05   8.46419807e+05   1.76734362e+06]


# In[ ]:


print(metrics.mean_squared_error(y_test, reg.predict(x_test)))
#58127859345.8


# In[ ]:


reg.score(x_test,y_test)
#0.6035


# Building a model with all parameters
# ------------------------------------

# In[ ]:


df=pd.get_dummies(data,columns=['waterfront','view','condition','grade','zipcode'], drop_first=True)


# In[ ]:


y=data['price']
df= df.drop(['date','id','price'],axis=1)


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(df,y,train_size=0.8,random_state=42)


# In[ ]:


reg.fit(x_train,y_train)


# In[ ]:


print('Coefficients: \n', reg.coef_)
print(metrics.mean_squared_error(y_test, reg.predict(x_test)))
print(reg.score(x_test,y_test))


#  - Mean Square Error: 24035065564.4
#  - R-square:  0.832372084112

# In[ ]:




