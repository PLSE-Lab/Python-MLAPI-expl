#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# 

# In[ ]:


train=pd.read_csv("../input/train.csv")
train.head(30)


# In[ ]:


train['Embarked'].unique()


# In[ ]:


train.info()


# In[ ]:


train.describe()


# **Exploratory Data Analysis**
# Let's begin some exploratory data analysis! We'll start by checking out missing data!
# 
# **Missing Data**
# We can use seaborn to create a simple heatmap to see where we are missing data!

# In[7]:


train.isnull().head()


# In[8]:


train.corr()


# In[9]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[10]:


sns.heatmap(train.corr(),annot=True)


# In[11]:


#sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Embarked',data=train,palette='Set3')#'RdBu_r')


# In[12]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


# In[13]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In[14]:


sns.distplot(train['Age'].dropna(),kde=True,color='darkred',bins=20)


# In[16]:


sns.countplot(x='SibSp',data=train)


# In[17]:


sns.countplot(x='Survived',hue='SibSp',data=train)


# In[18]:


train['Fare'].hist(color='green',bins=30,figsize=(8,4))


# **Data Cleaning**
# We want to fill in missing age data instead of just dropping the missing age data rows. One way to do this is by filling in the mean age of all the passengers (imputation). However we can be smarter about this and check the average age by passenger class.

# In[19]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='summer')


# We can see the wealthier passengers in the higher classes tend to be older, which makes sense. We'll use these average age values to impute based on Pclass for Age.

# In[20]:


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


# In[21]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# In[22]:


c=train[['Age','Pclass']]


# In[25]:


c.iloc[:,0]


# In[26]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Now Let's Drop the cabin columns and rows which has Nan data

# In[27]:


train.drop('Cabin',axis=1,inplace=True)
train.head()


# In[28]:


train.info()


# In[29]:


train.dropna(inplace=True)


# **Converting Categorical Features**
# We'll need to convert categorical features to dummy variables using pandas! Otherwise our machine learning algorithm won't be able to directly take in those features as inputs.

# In[30]:


sex = pd.get_dummies(train['Sex'],drop_first=True)


# In[31]:


embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[32]:


train = pd.concat([train,sex,embark],axis=1)
train.head()


# In[33]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train.head()


# In[34]:


train.info()


# Now Let's Start modelling our data
# 
# ****Building a Logistic Regression model**
# 
# * Train Test Split

# In[35]:


from sklearn.model_selection import train_test_split


# In[37]:


X = train.drop('Survived',axis=1)
y = train['Survived']
X.head()


# In[38]:


X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)


# **Training and Predicting**

# In[39]:


from sklearn.linear_model import LogisticRegression


# In[40]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[41]:


import pickle


# In[42]:


pickle_out=open("classifier.pickle","wb")
pickle.dump(logmodel,pickle_out)
pickle_out.close()


# In[43]:


pickle_in=open('classifier.pickle','rb')# rb read mode
cnn_classifier=pickle.load(pickle_in)


# In[44]:


cnn_classifier.predict(X_test)


# In[45]:


predictions = logmodel.predict(X_test)


# In[46]:


predictions


# **Evaluation**

# In[47]:


from sklearn.metrics import classification_report, confusion_matrix


# In[48]:


print(classification_report(y_test,predictions,digits=4))


# In[49]:


print(confusion_matrix(y_test,predictions))


# In[50]:


logmodel.coef_


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




