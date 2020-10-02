#!/usr/bin/env python
# coding: utf-8

# **Import the necessary libraries**

# In[ ]:



import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
  


# Read the necessary files

# In[ ]:


titanic_filepath = ('../input/titanic/train.csv')
titanic_data= pd.read_csv(titanic_filepath)
test_filepath = ('../input/titanic/test.csv')
test_data=pd.read_csv(test_filepath)


# In[ ]:


titanic_data.head()


# In[ ]:





# Select necessary features based on analysis

# In[ ]:


features=['Sex','Fare', 'Pclass','Parch','SibSp']


# Extract only the features

# In[ ]:


x=titanic_data[features]
test_x=test_data[features]


# 

# In[ ]:


y=titanic_data.Survived


# Observe that Sex is a not numerical value

# In[ ]:


x.head()


# In[ ]:


cleanup_nums = {"Sex":     {"male": 1, "female": 2}}
cleanup_nums2 = {"Embarked":     {"S": 1, "C": 2, "Q": 3}}
x.head()


# Replace the Sex values with 'cleanup_nums'

# In[ ]:


x.replace(cleanup_nums, inplace=True)
x.head()
test_x.replace(cleanup_nums, inplace=True)
x.replace(cleanup_nums2, inplace=True)
x.head()
test_x.replace(cleanup_nums2, inplace=True)
x.head()


# Fill blank cells with mean of data

# In[ ]:



x=x.fillna(x.mean())
test_x=test_x.fillna(test_x.mean())


# Define model and fit it to data

# In[ ]:


model=XGBClassifier()
model.fit(x,y)


# Get the sample submission file

# In[ ]:


submission_path = ('../input/titanic/gender_submission.csv')
submission= pd.read_csv(submission_path)


# Input predictions into survived column

# In[ ]:


submission['Survived']=model.predict(test_x)




# Replace PassengerId from test_data

# In[ ]:


submission['PassengerId']=test_data['PassengerId']


# In[ ]:


submission.columns=['PassengerId','Survived']


# Have a look to check if all if fine

# In[ ]:


submission.columns=['PassengerId','Survived']
submission.head()


# Convert it to a .csv file

# In[ ]:


submission.to_csv('Submission.csv', index=False)


# Submit it and hooray!!
# **Upvote if you liked it**
