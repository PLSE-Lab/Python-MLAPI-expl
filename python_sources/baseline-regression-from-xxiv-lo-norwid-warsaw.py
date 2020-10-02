#!/usr/bin/env python
# coding: utf-8

# ### Import bibliotek

# In[ ]:


import pandas as pd
import os
print(os.listdir("../input"))
pd.set_option('display.max_columns', 100) 
from sklearn.linear_model import LinearRegression as regression


# > ### Data Load

# In[ ]:


path_to_data = "../input/"


# In[ ]:


train = pd.read_csv(path_to_data + "train.csv")
test = pd.read_csv(path_to_data + "test.csv")


# In[ ]:


train.head()


# In[ ]:


train.shape, test.shape


# ### Feature engineering

# In[ ]:


#fill na with 0, not always optimal!
train.fillna(0, inplace = True)
test.fillna(0, inplace = True)


# In[ ]:


features = ["OverallQual", "GrLivArea", "GarageCars", "GarageArea", "TotalBsmtSF",
 "FullBath", "TotRmsAbvGrd", "YearBuilt"]
target = 'SalePrice'


# In[ ]:


X_train = train[features]
X_test = test[features]
y = train[target] 


# > ### Model training

# In[ ]:


nasz_model = regression().fit(X_train, y)


# In[ ]:


nasz_model


# In[ ]:


predictions = nasz_model.predict(X_test)


# In[ ]:


predictions[:10]


# In[ ]:


predictions_table = pd.read_csv(path_to_data + 'sample_submission.csv')
predictions_table.head()


# In[ ]:


predictions_table['SalePrice'] = predictions
predictions_table.head()


# ### Save predictions

# In[ ]:


predictions_table.to_csv('baseline_reg_sub.csv', index = False)
#team name: XXIV_LO_NORWID_xxx


# In[ ]:




