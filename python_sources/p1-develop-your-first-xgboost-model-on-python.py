#!/usr/bin/env python
# coding: utf-8

# # Load Libraries

# In[ ]:


import pandas as pd
import numpy as numpy
import xgboost as xgb #contains both XGBClassifier and XGBRegressor


# # Load Dataset 

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')


# In[ ]:


data.info()


# In[ ]:


#Get Target data 
y = data['Outcome']

#Load X Variables into a Pandas Dataframe with columns 
X = data.drop(['Outcome'], axis = 1)


# In[ ]:


X.head()


# # Check X Variables

# In[ ]:


#Check size of data
X.shape


# In[ ]:


X.isnull().sum()
#We do not have any missing values


# # Build Model

# In[ ]:


xgbModel = xgb.XGBClassifier() #max_depth=3, n_estimators=300, learning_rate=0.05


# In[ ]:


xgbModel.fit(X,y)


# # Check Accuracy

# In[ ]:


print (f'Accuracy - : {xgbModel.score(X,y):.3f}')


# # END
