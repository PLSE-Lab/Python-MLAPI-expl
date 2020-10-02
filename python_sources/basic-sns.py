#!/usr/bin/env python
# coding: utf-8

# ## messing about with _seaborn_
# 
# let's prepare data below, then analyze
# 

# In[ ]:


import numpy as np
import pandas as pd

import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#load & clean from @omarelgabry
train = train.drop(['PassengerId','Name','Ticket'], axis=1)
test    = test.drop(['Name','Ticket'], axis=1)
#impute median / mode values
train["Embarked"] = train["Embarked"].fillna("S")
train["Fare"].fillna(train["Fare"].median(), inplace=True)
train["Age"].fillna(train["Age"].median(), inplace=True)

#Print to standard output, and see the results in the "log" section below after running your script
#print(train.head())



print("\n\nSummary statistics of training data")
print(list(train.columns.values))


print(train.describe())


# ///////


# In[ ]:


sns.set_palette('pastel')

sns.factorplot('Embarked','Survived', data=train,size=4,aspect=3,kind="bar")


# In[ ]:


sns.factorplot('Pclass','Survived', data=train,size=4,aspect=3,kind="bar")


# In[ ]:


#convert cabin to bool & analyze as factor
#sns.factorplot('Cabin','Survived', data=train,size=4,aspect=3,kind="bar")


# In[ ]:


sns.lmplot(x='Fare', y='Survived', data=train, size=7, logistic=True)


# In[ ]:


sns.lmplot(x='Age', y='Survived', data=train, size=7, logistic=True) 



# In[ ]:


# //////

#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)

