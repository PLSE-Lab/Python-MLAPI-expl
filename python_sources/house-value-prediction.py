#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


import numpy as np


# In[ ]:


data = pd.read_csv('../input/kc-house-data/kc_house_data.csv',engine='python')


# In[ ]:





# In[ ]:


data


# In[ ]:


# Here we have year of built as well as year of renovation. So I am considering if the house is renovated then it will be the effective year.
# Also there are various values of sqft. So I am combining all of them into just one value.


# In[ ]:


p1 = data['yr_built'].to_numpy()


# In[ ]:


p2 = data['yr_renovated'].to_numpy()


# In[ ]:


for i in range(0,21613):
    if(p2[i]>p1[i]):
        p1[i]=p2[i]


# In[ ]:


p1


# In[ ]:


data['yr_effective'] = p1


# In[ ]:


data


# In[ ]:


data['sqft_total'] = data['sqft_living'] + data['sqft_lot'] + data['sqft_above'] + data['sqft_basement'] + data['sqft_living15'] + data['sqft_lot15']
data['bathrooms'] = data['bathrooms'].astype(int)


# In[ ]:


data


# In[ ]:


#Lets split out only required data
dataSub = data[['price','bedrooms','bathrooms','floors','waterfront','view','sqft_total','yr_effective','lat','long']]


# In[ ]:


dataSub.head(5)


# In[ ]:


dataSub.describe()


# In[ ]:


dataSub.isnull().sum()


# In[ ]:


dataSub.dtypes


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


# We will use DecisionTree here rather than Linear Regression


# In[ ]:


from sklearn.tree import DecisionTreeRegressor


# In[ ]:


x = dataSub.iloc[:,1:10]
x


# In[ ]:


import seaborn as sns

correlation = dataSub.corr()
sns.heatmap (correlation,annot=True)


# In[ ]:


y = dataSub['price']
y


# In[ ]:


#lets split into train and test now.


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=30)


# In[ ]:


reg = DecisionTreeRegressor()


# In[ ]:


#Lets fit the model now


# In[ ]:


reg.fit(x_train,y_train)


# In[ ]:


pred = reg.predict(x_test)
pred


# In[ ]:


y_test


# In[ ]:


# Above results are almost similar , example:
# pred value = 379950 , actual = 384950
# pred value = 290000 , actual = 302400
# pred value = 550000 , actual = 557000
#Lets plot and see

ax1 = sns.distplot(y_test, hist=False, color='r', label='Actual')
sns.distplot(pred, hist=False, color='b', label='pred', ax=ax1)


# In[ ]:


#lets manually enter the values of entry #2 and check
x = [['3','2','2','0','0','21711','1991','47.7210','-122.319']]
reg.predict(x)
#and the actual value in table was 538000


# In[ ]:


# I feel I have done a good work on this data


# In[ ]:


# Feel free to give your reviews on it


# In[ ]:


# Please let me know if I had done any mistake or how I could make it better


# In[ ]:


# I am attaching the data along with the post 


# In[ ]:


# Do try this dataset, this is interesting 


# In[ ]:




