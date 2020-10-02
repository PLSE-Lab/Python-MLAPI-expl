#!/usr/bin/env python
# coding: utf-8

# # // Decision Tree Regressor //

# In[3]:


import pandas as pd


# In[4]:


from sklearn.tree import DecisionTreeRegressor


# In[5]:


file_path = '../input/home-data-for-ml-course/train.csv'


# In[6]:


home_data = pd.read_csv(file_path)


# In[7]:


y = home_data.SalePrice


# In[8]:


features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X= home_data[features]


# In[9]:


y


# In[ ]:


X


# In[ ]:


model = DecisionTreeRegressor()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


# In[ ]:


model.fit(train_X, train_y)


# In[ ]:


predictions = model.predict(val_X)


# In[ ]:


predictions


# In[ ]:


from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(val_y,predictions))

