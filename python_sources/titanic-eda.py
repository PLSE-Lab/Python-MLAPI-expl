#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
#invite people for the Kaggle party
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import math
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# read training and test data.

# In[ ]:


df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")


# training data sample

# In[ ]:


df_train.head()


# test data sample

# In[ ]:


df_test.head()


# show how many data are missing

# In[ ]:


df_train.isnull().sum()


# we can see that a lot of the cabin numbers values are missing we'll remove this column as it probably won't matter in the survival rate

# we check column types

# In[ ]:


df_train.dtypes


# we fix the type for some columns

# In[ ]:


df_train.Survived = df_train.Survived.astype('category')
df_train.Pclass = df_train.Pclass.astype('category')
df_train.SibSp = df_train.SibSp.astype('category')
df_train.Parch = df_train.Parch.astype('category')


# descriptive statistics summary for our class 'Survived'

# In[ ]:


sns.countplot('Survived',data=df_train)
plt.show()


# we'll remove weak attrbuites

# In[ ]:


df_train = df_train.drop(columns="PassengerId")
df_train = df_train.drop(columns="Cabin")


# now we'll fill the missing values for the age 

# In[ ]:


df_train.fillna(df_train.mean(), inplace=True)
corrmat = df_train.corr().abs()


# we can understand more using more plots

# In[ ]:


plt.ylim(0, 100)
sns.boxplot(data=df_train,  y="Fare", x = "Survived")


# we can see that the higher the fare the higher the survival rate

# In[ ]:


plt.ylim(0, 60)
sns.boxplot(data=df_train,  y="Age", x = "Survived")


# we can notice that survivals have lower Q1 meaning that younger people had higher survival rate

# In[ ]:


pd.crosstab(df_train.Survived, df_train.Pclass)


# we notice people with in 1st class had higher survival rate while people in 3rd class had lower survival rates

# In[ ]:


pd.crosstab(df_train.Survived, df_train.Parch)


# In[ ]:


pd.crosstab(df_train.Survived, df_train.SibSp)


# we see that people with one or two family member had better survival rate and more or less had lower

# In[ ]:


pd.crosstab(df_train.Survived, df_train.Sex)


# it's very obvious that the survival rate if women with much more high than that of men
