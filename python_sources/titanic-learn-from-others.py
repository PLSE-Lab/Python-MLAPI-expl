#!/usr/bin/env python
# coding: utf-8

# Start

# In[ ]:


# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# In[ ]:


# get titanic & test csv files as a DataFrame
titanic_df = pd.read_csv("../input/train.csv")
test_df    = pd.read_csv("../input/test.csv")

titanic_df.head()


# In[ ]:


titanic_df.info()
print('-----------------')
test_df.info()


# In[ ]:


# drop unnecessary columns, these columns won't be useful in analysis and prediction

titanic_df.drop(['PassengerId','Name','Ticket'], axis=1, inplace=True)
test_df.drop(['PassengerId','Name','Ticket'], axis=1, inplace=True)


# In[ ]:


titanic_df['Embarked'].fillna("S", inplace=True)

sns.factorplot('Embarked', 'Survived', data=titanic_df, size=3, aspect=2)


# In[ ]:


print(titanic_df['Embarked'].unique())
print(titanic_df['Embarked'].describe())


# In[ ]:




