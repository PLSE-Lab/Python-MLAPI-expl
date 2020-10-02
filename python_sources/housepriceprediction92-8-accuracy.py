#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:



import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.ensemble import GradientBoostingRegressor


# ## Reading Data

# In[ ]:


Housedata = pd.read_csv("../input/kc_house_data.csv")
Housedata.head()


# ## Feature Engineering 

# In[ ]:


Housedata = Housedata.drop(['id'],axis =1)
Housedata['year_selling']=pd.to_datetime(Housedata['date']).apply(lambda s: s.year)
Housedata['age']=Housedata['year_selling']-Housedata['yr_built']


# In[ ]:


del Housedata['date']
del Housedata['yr_built']


# ## Correlation Matrix

# In[ ]:


CorrelationMatrix = Housedata.corr()
fig = plt.figure(figsize = (15,15))
sns.heatmap(CorrelationMatrix,annot = True,cmap="coolwarm")
plt.show()


# ## No features are highly correlated so will keep all features

# ## Data Visualization using bar plot ,distribution of bedrooms ,bathrooms and number of floors

# In[ ]:


Housedata['bedrooms'].value_counts().plot(kind='bar')
plt.title('Count of Bedrooms')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Count')
sns.despine


# In[ ]:



Housedata['bathrooms'].value_counts().plot(kind='bar')
plt.title('Count of Bathrooms')
plt.xlabel('Number of Bathrooms')
plt.ylabel('Count')
sns.despine


# In[ ]:


Housedata['floors'].value_counts().plot(kind='bar')
plt.title('Count of floors')
plt.xlabel('Number of floors')
plt.ylabel('Count')
sns.despine


# ## Scatterplot between sqft_living vs price,bedrooms vs price ,waterfront vs price
# 

# In[ ]:


plt.scatter(Housedata.sqft_living,Housedata.price)
plt.title("Square Feet vs Price ")


# In[ ]:


plt.scatter(Housedata.bedrooms,Housedata.price)
plt.title("Bedroom and Price ")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.show()
sns.despine


# In[ ]:


plt.scatter(Housedata.waterfront,Housedata.price)
plt.title("Waterfront vs Price ( 0= no waterfront)")


# ## Seperate houseprice  and input features from dataset

# In[ ]:


Houseprice  = Housedata['price']
del  Housedata['price']


# In[ ]:


Inputfeatures = Housedata.values


# In[ ]:


Houseprice.values


# ## Split data into training and testing

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test ,Y_train ,Y_test =train_test_split(Inputfeatures ,Houseprice, test_size = 0.10,random_state =2)


# ## Build XGBoost Regression model

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
GBR = GradientBoostingRegressor(n_estimators=2000, max_depth=5,verbose=2,learning_rate=0.05, validation_fraction=0.1,random_state=2)
GBR


# ## Train model on training data

# In[ ]:


GBR.fit(X_train, Y_train)


# In[ ]:


print("Accuracy on training data --> ", GBR.score(X_train, Y_train)*100)


# ## Accuracy on training data is 98.6%

# In[ ]:


print("Accuracy on test data ", GBR.score(X_test, Y_test)*100)


# ## Accuracy on test data is 92.8%

# In[ ]:


PredictedpriceGBR = GBR.predict(X_test)
Predictedprice = pd.Series(PredictedpriceGBR)
Predictedprice[0:10]


# In[ ]:



RealPrice = Y_test
RealPrice[0:10]


# In[ ]:


RealPrice = Y_test
RealPrice=pd.Series(RealPrice)
RealPrice[0:10]


# In[ ]:


DataframePredictedandRealPrice= pd.DataFrame({ 'RealPrice':RealPrice[0:10],'Predictedprice':PredictedpriceGBR[0:10]})
DataframePredictedandRealPrice


# ## Graph of real and predicted price for first 10 entries

# In[ ]:


ax = DataframePredictedandRealPrice.plot.bar()
plt.xticks(rotation =90)
plt.title('House Price Prediction using Gradient Boosting Regressor',pad = 20,fontsize = 15)

