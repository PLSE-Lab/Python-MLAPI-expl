#!/usr/bin/env python
# coding: utf-8

# # Load Libraries

# In[ ]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# # Load Dataset 

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')


# # Check Data for any missing values

# In[ ]:


data.head()


# In[ ]:


data.isnull().sum()


# In[ ]:


#Get Target data 
y = data['target']

#Load X Variables into a Pandas Dataframe with columns 
X = data.drop(['target'], axis = 1)


# In[ ]:


print(f'X : {X.shape}')


# # Divide Data into Train and test

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)


# In[ ]:


print(f'X_train : {X_train.shape}')
print(f'y_train : {y_train.shape}')
print(f'X_test : {X_test.shape}')
print(f'y_test : {y_test.shape}')


# # Build Basic Random Forest Model

# In[ ]:


rf_Model = RandomForestClassifier()


# In[ ]:


rf_Model.fit(X_train,y_train)


# # Check Accuracy

# In[ ]:


print (f'Train Accuracy - : {rf_Model.score(X_train,y_train):.3f}')
print (f'Test Accuracy - : {rf_Model.score(X_test,y_test):.3f}')


# # END
