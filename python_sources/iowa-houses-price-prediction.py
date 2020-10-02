#!/usr/bin/env python
# coding: utf-8

# # Introduction
# **This will be your workspace for the [Machine Learning course](https://www.kaggle.com/learn/machine-learning).**
# 
# You will need to translate the concepts to work with the data in this notebook, the Iowa data. Each page in the Machine Learning course includes instructions for what code to write at that step in the course.
# 
# # Write Your Code Below

# In[1]:


import pandas as pd

Iowa_file_path = '../input/house-prices-advanced-regression-techniques/train.csv'
Iowa_data = pd.read_csv(Iowa_file_path)
#Iowa_data.head()
print(Iowa_data.describe())


# In[2]:


Columns_list = Iowa_data.columns
print(Columns_list)


# In[3]:


Iowa_price_data = Iowa_data.SalePrice
Iowa_price_data.head()


# In[4]:


columns_of_interst = ['TotalBsmtSF','SalePrice']
two_columns_of_data = Iowa_data[columns_of_interst]
two_columns_of_data.describe()


# In[8]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
y = Iowa_data.SalePrice
Imp_features = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X = Iowa_data[Imp_features]
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
Iowa_model = DecisionTreeRegressor()
Iowa_model.fit(train_X,train_y)
predicted_home_prices = Iowa_model.predict(val_X)
print(mean_absolute_error(val_y, predicted_home_prices))
print(r2_score(val_y, predicted_home_prices))



# In[17]:


from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np

def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)
my_mae = []
node_list =  [i for i in range(2,200)]
for max_leaf_nodes in node_list:
    my_mae.append(get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y))
    #print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
#index = node_list.index(min(my_mae)) 
#print(node_list[my_mae.index(min(my_mae))])
plt.plot(node_list, my_mae, color='red')
plt.show()
    


# In[13]:


print(Iowa_data.isnull().sum().sort_values(ascending=False).head(15))
Iowa_data.shape


# In[18]:


New_data_after_dropping = Iowa_data.dropna(axis=1)
New_data_after_dropping.shape


# # Bellow expression is generator expression 

# In[19]:


Column_with_missing_value = (col for col in Iowa_data.columns if Iowa_data[col].isnull().any())
for col in Column_with_missing_value :
    print(col)


# In[20]:


cols_with_missing = [col for col in Iowa_data.columns 
                                 if Iowa_data[col].isnull().any()]
redued_Iowa_data = Iowa_data.drop(cols_with_missing, axis=1)
print(cols_with_missing)


# 
# **If you have any questions or hit any problems, come to the [Learn Discussion](https://www.kaggle.com/learn-forum) for help. **
# 
# **Return to [ML Course Index](https://www.kaggle.com/learn/machine-learning)**

# 
