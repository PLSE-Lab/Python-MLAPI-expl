#!/usr/bin/env python
# coding: utf-8

# # In this notebook we will build a Logistic Regression model to predict survival of the Titanic

# # Step 1: Import Libraries
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# #Step 2: Import Data

# In[ ]:


train = pd.read_csv('../input/train.csv')


# In[ ]:


train.head()


# #Step Three: Fill Missing Data

# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False, cmap='inferno')
#Visualization of Missing Data


# In[ ]:


train.describe()


# In[ ]:


corr = train.corr()
f, ax = plt.subplots(figsize=(10,6))
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)

sns.heatmap(corr, cmap='bwr', linewidths=0.1,vmax=1.0, square=True, annot=True)


# In[ ]:


train.groupby('Pclass').mean()
# We need to fill in Age values for passengers.  Pclass is the variable which is 
#most correlated with it, so it will be used to help fill in the age column.


# In[ ]:


# We took the values from the group.by statement to fill in missing age values, based
#on their passenger class.
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 38.23

        elif Pclass == 2:
            return 29.88

        else:
            return 25.14

    else:
        return Age


# In[ ]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False, cmap='inferno')
#Visualization of Missing Data


# In[ ]:


train.drop('Cabin',axis=1,inplace=True)


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False, cmap='inferno')
#Visualization of Missing Data


# In[ ]:


train.dropna(inplace=True)


# #Step 4: Create Dummy Variables for Categorical Data

# In[ ]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[ ]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


train = pd.concat([train,sex,embark],axis=1)


# In[ ]:


train.head()


# #Step 5: Import and Run Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)


# #Step 6: Import, Instantiate and Run the Logistic Regression 

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


predictions = logmodel.predict(X_test)


# #Step 7: Evaluate the Model

# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,predictions))


# # Congratulations, we are done!
