#!/usr/bin/env python
# coding: utf-8

# In[40]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Import Libraries**

# In[41]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# **The Data**

# In[42]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[43]:


train.head()


# In[44]:


test.head()


# **EXPLORATORY DATA ANALYSIS AND DATA CLEANING**

# In[45]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[46]:


sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[47]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# In[48]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=test,palette='winter')


# In[49]:


#Replace age of Pclass 1, Pclass 2, Pclass 3 with 37, 29, 24 respectively in train data
#Replace age of Pclass 1, Pclass 2, Pclass 3 with 42, 26, 24 respectively in test data
def impute_age_train(cols):
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
    
    
    
def impute_age_test(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 42

        elif Pclass == 2:
            return 26

        else:
            return 24

    else:
        return Age


# In[50]:


train['Age'] = train[['Age','Pclass']].apply(impute_age_train,axis=1)
test['Age'] = test[['Age','Pclass']].apply(impute_age_test,axis=1)


# In[51]:


#Lets drop 'Cabin' column and row with some missing values
train.drop('Cabin',axis=1,inplace=True)
test.drop('Cabin',axis=1,inplace=True)

train.dropna(inplace=True)
test.fillna(test.mean(),inplace=True)


# In[52]:


train.info()


# In[53]:


test.info()


# In[54]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train)


# In[55]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


# In[56]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In[57]:


sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)


# In[58]:


train['Age'].hist(bins=30,color='darkred',alpha=0.7)


# In[59]:


sns.countplot(x='SibSp',data=train)


# In[60]:


train['Fare'].hist(color='green',bins=40,figsize=(8,4))


# **CONVERTING CATEGORICAL FEATURES**

# In[61]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)
train.head()


# In[62]:


sex = pd.get_dummies(test['Sex'],drop_first=True)
embark = pd.get_dummies(test['Embarked'],drop_first=True)
test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
test = pd.concat([test,sex,embark],axis=1)
test.head()


# **SPLIT TRAIN DATA INTO TRAIN AND TEST SET**
# 

# In[63]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)


# **TRAINING AND CHECKING ACCURACY WITH DIFFERENT MODELS**

# 1. LOGISTIC REGRESSION

# In[64]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
pred_logmodel = logmodel.predict(X_test)


# 2. KNN MODEL

# In[65]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=25)
knn.fit(X_train,y_train)
pred_knn = knn.predict(X_test)


# 3. DECISION TREE CLASSIFIER

# In[66]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
pred_dtree = dtree.predict(X_test)


# 4. RANDOM FOREST CLASSIFIER

# In[67]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=300)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)


# **EVALUATION OF DIFFERENT MODELS**

# In[68]:


from sklearn.metrics import classification_report,confusion_matrix


# In[69]:


#Logistic Regression
print(classification_report(y_test,pred_logmodel))
print(confusion_matrix(y_test,pred_logmodel))


# In[70]:


#KNN Model
print(classification_report(y_test,pred_knn))
print(confusion_matrix(y_test,pred_knn))


# In[71]:


#Decision Tree Classifier
print(classification_report(y_test,pred_dtree))
print(confusion_matrix(y_test,pred_dtree))


# In[72]:


#Random Forest Classifier
print(classification_report(y_test,pred_rfc))
print(confusion_matrix(y_test,pred_rfc))


# In[73]:


#Since best accuracy is coming from Random Forest Classifier(84%), we would take that.
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=300)
rfc.fit(train.drop('Survived',axis=1), train['Survived'])
pred_rfc = rfc.predict(test) 


# In[78]:


submission= pd.DataFrame({
    "PassengerId" : test["PassengerId"],
    'Survived' : pred_rfc
})

submission.to_csv('submission.csv',index=False)


# In[ ]:




