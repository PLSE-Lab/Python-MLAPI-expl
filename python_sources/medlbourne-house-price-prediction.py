#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing

melbourne_file_path = '../input/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path); 
melbourne_data.describe()


# In[ ]:


melbourne_data.columns


# In[ ]:


y = melbourne_data.Price


# In[ ]:


melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']


# In[ ]:


x = melbourne_data[melbourne_features]
x.describe()


# In[ ]:


x.head()


# In[ ]:


from sklearn.tree import DecisionTreeRegressor

melbourne_model = DecisionTreeRegressor(random_state=1)
melbourne_model.fit(x, y)


# In[ ]:


print("Making predictions for the following 5 houses:")
print(x.head())
print("The predictions are")
print(melbourne_model.predict(x.head()))

