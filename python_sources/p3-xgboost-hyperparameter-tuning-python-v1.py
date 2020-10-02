#!/usr/bin/env python
# coding: utf-8

# # Load Libraries

# In[ ]:


import pandas as pd
import numpy as numpy
import xgboost as xgb #contains both XGBClassifier and XGBRegressor


# # Load Dataset 

# In[ ]:


data = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')


# In[ ]:


#Get Target data 
y = data['Outcome']

#Load X Variables into a Pandas Dataframe with columns 
X = data.drop(['Outcome'], axis = 1)


# # Divide Data into Train and Test 

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)


# In[ ]:


print(f'X_train : {X_train.shape}')
print(f'y_train : {y_train.shape}')
print(f'X_test : {X_test.shape}')
print(f'y_test : {y_test.shape}')


# # Build Model with Tuning

# # Trial 1 - Learning Rate (0.1)

# In[ ]:


xgbModel = xgb.XGBClassifier(learning_rate =0.1, n_estimators=1000, max_depth=5)
xgbModel.fit(X_train,y_train,early_stopping_rounds=30, 
             eval_set=[(X_test, y_test)], verbose=False )


# ## Trail 1 - Accuracy

# In[ ]:


print (f'Train Accuracy - : {xgbModel.score(X_train,y_train):.3f}')
print (f'Test Accuracy - : {xgbModel.score(X_test,y_test):.3f}')


# ## Trial 1 - Slow Learning Rate (0.05)

# In[ ]:


xgbModel = xgb.XGBClassifier(learning_rate =0.05, n_estimators=1000, max_depth=5)
xgbModel.fit(X_train,y_train,early_stopping_rounds=30, 
             eval_set=[(X_test, y_test)], verbose=False )


# In[ ]:


print (f'Train Accuracy - : {xgbModel.score(X_train,y_train):.3f}')
print (f'Test Accuracy - : {xgbModel.score(X_test,y_test):.3f}')


# # END

# More on Hyperparameter Tuning - https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
