#!/usr/bin/env python
# coding: utf-8

# # Introduction
# **This will be your workspace for the [Machine Learning course](https://www.kaggle.com/learn/machine-learning).**
# 
# You will need to translate the concepts to work with the data in this notebook, the Iowa data. Each page in the Machine Learning course includes instructions for what code to write at that step in the course.
# 
# # Write Your Code Below

# In[ ]:


import pandas as pd

main_file_path = '../input/house-prices-advanced-regression-techniques/train.csv' # this is the path to the Iowa data that you will use
data = pd.read_csv(main_file_path)

# Run this code block with the control-enter keys on your keyboard. Or click the blue botton on the left
print('Some output from running this cell')


# In[ ]:


data.describe()


# In[ ]:


data.columns


# In[ ]:


data_price=data.SalePrice


# In[ ]:


two_columns=['Fence','SaleType']


# In[ ]:


data_of_two=data[two_columns]
data_of_two.describe()


# In[ ]:


y=data.SalePrice


# In[ ]:


usa_predictors=['YearBuilt', 'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',  '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr',  
        'GarageCars', 'GarageArea', 
     'WoodDeckSF', 'OpenPorchSF',
        'PoolArea']


# In[ ]:


x=data[usa_predictors]


# In[ ]:



x.info()


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
usa_model=DecisionTreeRegressor()
usa_model.fit(x,y)


# In[ ]:


print("Making prediction for 5 houses")
print(x.head())
print(usa_model.predict(x.head()))


# In[ ]:


from sklearn.metrics import mean_absolute_error
predict_homes_price=usa_model.predict(x)
mean_absolute_error(y,predict_homes_price)


# In[ ]:


from  sklearn.model_selection import train_test_split
train_x,val_x,train_y,val_y=train_test_split(x,y,random_state=0)
usa_model.fit(train_x,train_y)


# In[ ]:


val_predict=usa_model.predict(val_x)


# In[ ]:


print(mean_absolute_error(val_y,val_predict))


# In[ ]:


def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)


# In[ ]:


for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# 
# **If you have any questions or hit any problems, come to the [Learn Discussion](https://www.kaggle.com/learn-forum) for help. **
# 
# **Return to [ML Course Index](https://www.kaggle.com/learn/machine-learning)**

# 
