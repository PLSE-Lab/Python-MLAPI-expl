#!/usr/bin/env python
# coding: utf-8

# # Load Libraries

# In[ ]:


import pandas as pd
from sklearn.linear_model import LogisticRegression


# # Load Dataset 

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')


# In[ ]:


data.info()


# In[ ]:


#Get Target data 
y = data['diagnosis']

#Load X Variables into a Pandas Dataframe with columns 
X = data.drop(['id','diagnosis','Unnamed: 32'], axis = 1)


# In[ ]:


X.head()


# # Check X Variables

# In[ ]:


#Check size of data
X.shape


# In[ ]:


X.isnull().sum()
#We do not have any missing values


# # Build Logistic Regression

# In[ ]:


logModel = LogisticRegression(max_iter=5000)


# In[ ]:


logModel.fit(X,y)


# # Check Accuracy

# In[ ]:


print (f'Accuracy - : {logModel.score(X,y):.3f}')


# # END
