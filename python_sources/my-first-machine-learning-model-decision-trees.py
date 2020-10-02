#!/usr/bin/env python
# coding: utf-8

# # Intro
# **This is your workspace for Kaggle's Machine Learning course**
# 
# You will build and continually improve a model to predict housing prices as you work through each tutorial.
# 
# The tutorials you read use data from Melbourne. The Melbourne data is not available in this workspace.  Instead, you will translate the concepts to work with the data in this notebook, the Iowa data.
# 
# # Write Your Code Below
# 

# In[9]:


import pandas as pd

main_file_path = '../input/train.csv' # this is the path to the Iowa data that you will use
data = pd.read_csv(main_file_path)

# The cod below will help you see how output appears when you run a code block
# Run this code block with the control-enter keys on your keyboard. Or click the blue botton on the left
print('hello world')
print(data.columns)


# 
# **If you have any questions or hit any problems, come to the [Learn Discussion](https://www.kaggle.com/learn-forum) for help. **
# 
# **Return to [ML Course Index](https://www.kaggle.com/learn/machine-learning)**

# In[10]:


two_cols = data[['SaleCondition','SalePrice']]
SP = data['SalePrice']


# In[11]:


y = SP
predictors = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X = data[predictors]
from sklearn.tree import DecisionTreeRegressor
IowaModel = DecisionTreeRegressor()

IowaModel.fit(X,y)
print('Making predictions with the model')
print(IowaModel.predict(X.head()))


# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X,y,random_state=0)
IowaModel.fit(X_train,y_train)
val_predictions = IowaModel.predict(X_val)
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_val,val_predictions)


# In[13]:


def get_error(leaf_nodes,X_train,X_val,y_train,y_val):
    IowaModel = DecisionTreeRegressor(max_leaf_nodes=leaf_nodes)
    IowaModel.fit(X_train,y_train)
    prediction = IowaModel.predict(X_val)
    error = mean_absolute_error(y_val,prediction)
    return error
for x in [5,50,500,5000]:
    my_error = get_error(x,X_train,X_val,y_train,y_val)
    print('For leaf node: ',x,', error is ',my_error)


# In[14]:


from sklearn.ensemble import RandomForestRegressor
RForest = RandomForestRegressor()
RForest.fit(X_train,y_train)
Prediction_forests = RForest.predict(X_val)
mean_absolute_error(y_val,Prediction_forests)


# In[15]:


# Read the test data
test = pd.read_csv('../input/test.csv')
# Treat the test data in the same way as training data. In this case, pull same columns.
X_test = test[predictors]
# Use the model to make predictions
predicted_prices = RForest.predict(X_test)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)




my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:




