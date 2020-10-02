#!/usr/bin/env python
# coding: utf-8

# Import Libraries
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


#  Reading in the data files into a pandas dataframe

# In[2]:


training_df = pd.read_csv('../input/train.csv')
testing_df=pd.read_csv('../input/test.csv')


# In[3]:


training_df.head()


# In[4]:


testing_df.head()


# ##Data Visualiation
# heatmap for missing data using Seaborn

# In[5]:


sns.heatmap(training_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[6]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=training_df)


# In[7]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=training_df)


# In[8]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=training_df)


# In[9]:


sns.distplot(training_df['Age'].dropna(),kde=False,bins=30)


# In[10]:


training_df['Age'].hist(bins=30,color='darkred',alpha=0.7)


# In[11]:


sns.countplot(x='SibSp',data=training_df)


# Data Cleaning

# In[12]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=training_df)


# Rich passengers in the higher classes tend to be older.Impute based on Pclass for Age.

# In[13]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


# In[14]:


training_df['Age'] = training_df[['Age','Pclass']].apply(impute_age,axis=1)


# In[15]:


sns.heatmap(training_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# drop the Cabin column and the row in Embarked that is NaN.

# In[16]:


training_df.drop('Cabin',axis=1,inplace=True)


# In[17]:


training_df.head()


# In[18]:


training_df.dropna(inplace=True)


# ## Converting Categorical Features 

# In[21]:


training_df.info()


# In[22]:


sex = pd.get_dummies(training_df['Sex'],drop_first=True)
embark = pd.get_dummies(training_df['Embarked'],drop_first=True)


# In[23]:


training_df.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[24]:


training_df = pd.concat([training_df,sex,embark],axis=1)


# In[25]:


training_df.head()


# Split the data andLogistic Regression model

# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(training_df.drop('Survived',axis=1), 
                                                    training_df['Survived'], test_size=0.30)


# Training and Predicting

# In[29]:


from sklearn.linear_model import LogisticRegression


# In[30]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[31]:


predictions = logmodel.predict(X_test)


# In[32]:


from sklearn.metrics import classification_report


# In[33]:


print(classification_report(y_test,predictions))


# In[ ]:




