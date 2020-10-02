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


# # Build Model

# In[ ]:


xgbModel = xgb.XGBClassifier() #max_depth=3, n_estimators=300, learning_rate=0.05


# In[ ]:


xgbModel.fit(X_train,y_train)


# # Predict on Test Data Set

# In[ ]:


test_pred = xgbModel.predict(X_test)
print(test_pred)


# # Combine Prediction with Test Actuals

# In[ ]:


Test_Actual_Pred = pd.DataFrame({ 'Actual' : y_test, 'Prediction': test_pred})
Test_Actual_Pred.head()


# # Check Accuracy

# In[ ]:


print (f'Train Accuracy - : {xgbModel.score(X_train,y_train):.3f}')
print (f'Test Accuracy - : {xgbModel.score(X_test,y_test):.3f}')


# # END
