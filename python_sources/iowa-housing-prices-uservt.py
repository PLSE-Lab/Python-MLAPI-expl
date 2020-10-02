#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Predict Iowa Housing Prices
# 
# 

# In[ ]:


import pandas as pd

iowahsng = '../input/iowa-house-prices/train.csv'
data = pd.read_csv(iowahsng)
data.head()


# In[ ]:


data.info()


# In[ ]:


data.columns


# In[ ]:


data.SalePrice.head()


# In[ ]:


sel_col = ['SalePrice', 'YearBuilt', 'LotArea', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
iowadf = data[sel_col]
iowadf.head()


# In[ ]:


iowadf.isnull().sum()


# In[ ]:


y = iowadf['SalePrice']
X = iowadf.drop('SalePrice', axis = 1)

from sklearn.model_selection import train_test_split

X_Train, X_Test, y_Train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
X_Train.shape


# In[ ]:


X_Test.shape


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
dtreg = DecisionTreeRegressor()


# In[ ]:


dtreg.fit(X_Train, y_Train)


# In[ ]:


pred = dtreg.predict(X_Test)


# In[ ]:


from sklearn import metrics


# In[ ]:


MAE = metrics.mean_absolute_error(y_test, pred)
MAE


# In[ ]:


def getmae(max_leafnodes, X_Train, X_Test, y_train, y_test):
    model = DecisionTreeRegressor(max_leaf_nodes= max_leafnodes, random_state=0)
    model.fit(X_Train, y_Train)
    pred_val = model.predict(X_Test)
    mae = metrics.mean_absolute_error(y_test, pred_val)
    return(mae)
    
    


# In[ ]:


for max_leafnode in [5,50,100,500,1000]:
    maeval = getmae(max_leafnode, X_Train, X_Test, y_Train, y_test)
    print ('Max Leaf nodes = {} and MAE vale = {}'.format(max_leafnode, maeval))


# In[ ]:


#get predictions using max_Leaf_nodes = 50

dcRegobj = DecisionTreeRegressor(max_leaf_nodes=50)


# In[ ]:


dcRegobj.fit(X_Train, y_Train)


# In[ ]:


pred_Val = dcRegobj.predict(X_Test)
MAEval = metrics.mean_absolute_error(y_test, pred_Val)
MAEval


# In[ ]:


# Using randomForest
from sklearn.ensemble import RandomForestRegressor
rfreg = RandomForestRegressor()


# In[ ]:


rfreg.fit(X_Train, y_Train)


# In[ ]:


rf_pred = rfreg.predict(X_Test)


# In[ ]:


mae = metrics.mean_absolute_error(y_test, rf_pred)
print('MAE with Random Forest method ={}'.format(mae))


# In[ ]:


#Try to optimize Random Forest mae value
X_Test.index


# In[ ]:


sub_df = pd.DataFrame(rf_pred, columns=['Predicted Values'])
sub_df.head()


# In[ ]:


sub_df.to_csv('iowa_submissionfile.csv', index = False)


# In[ ]:




