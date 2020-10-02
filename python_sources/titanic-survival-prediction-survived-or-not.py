#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ### import libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# ### reading our train and test dataset

# In[ ]:


train=pd.read_csv('../input/titanic/train.csv')
test=pd.read_csv('../input/titanic/test.csv')


# **lets check first five rows of dataset**

# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


print('Training dataset size : ',train.shape)
print('Testing dataset size : ',test.shape)


# In[ ]:


print('Null value is train dataset :\n',train.isnull().sum())


# In[ ]:


print('Null value is test dataset :\n',test.isnull().sum())


# # Data Visualization

# In[ ]:


fig=plt.figure(figsize=(12,8))
fig=sns.heatmap(train.isnull())


# In[ ]:


col=[col for col in train.columns if train[col].isnull().sum()>0]
for x in col:
    print('{} missing value : {} %'.format(x,np.round(train[x].isnull().mean(),3)))


# #### As we can see Cabin column has around 77% missing value so we will just drop this column 

# In[ ]:


fig=plt.figure(figsize=(12,8))
fig=sns.heatmap(test.isnull())


# In[ ]:


col=[col for col in test.columns if test[col].isnull().sum()>0]
for x in col:
    print('{} missing value : {} %'.format(x,np.round(test[x].isnull().mean(),3)))


# In[ ]:


train.drop('Cabin',axis=1,inplace=True)
test.drop('Cabin',axis=1,inplace=True)


# In[ ]:


sns.countplot(x='Survived',data=train)


# In[ ]:


sns.countplot(x='Survived',hue='Sex',data=train)


# ### No of female survivor is much more then the male survivor 

# In[ ]:


sns.countplot(x='Pclass',data=train)


# ### it clearly show that more people in pclass 3 compare to other class because its more cheap lets check the survival rate of different Pcalss 

# In[ ]:


sns.countplot(x='Pclass',hue='Survived',data=train)


# In[ ]:


train['Age'].plot.hist(bins=30)


# In[ ]:


train['Fare'].plot.hist(bins=40)


# ## Data Preprocessing 

# ### lets first handle age column because it has some nan value

# In[ ]:


train['Age']=train['Age'].fillna(train['Age'].mean())
test['Age']=test['Age'].fillna(test['Age'].mean())


# ### Handling categorical features

# In[ ]:


sex_train=pd.get_dummies(train["Sex"],drop_first=True)
embarked_train=pd.get_dummies(train["Embarked"],drop_first=True)

sex_test=pd.get_dummies(test["Sex"],drop_first=True)
embarked_test=pd.get_dummies(test["Embarked"],drop_first=True)


# In[ ]:


train=pd.concat([train,sex_train,embarked_train],axis=1)
train.drop(['PassengerId','Sex','Embarked','Name','Ticket'],axis=1,inplace=True)



test=pd.concat([test,sex_test,embarked_test],axis=1)
test_id=test['PassengerId']  # we will use it at data submission time
test.drop(['PassengerId','Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


test_id


# In[ ]:


train.head(2)


# In[ ]:


test['Fare']=test['Fare'].fillna(test['Fare'].mean())


# In[ ]:


test.head(2)


# ### Now its looks good

# In[ ]:


X=train.drop(['Survived'],axis=1)
y=train['Survived']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)


# In[ ]:


sc=StandardScaler()
X_scaled=sc.fit_transform(X_train)


# In[ ]:


test_scaled=sc.transform(test)


# ## KNeighbourClassifier 

# In[ ]:


knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled,y_train)
t_scaled=sc.transform(X_test)
p_test=knn.predict(t_scaled)
print('accuracy :',accuracy_score(y_test,p_test))


# ## RandomForestClassifier 

# In[ ]:


model = RandomForestClassifier(criterion= 'entropy',n_estimators=500,
                               bootstrap = True,
                               max_features = 'sqrt',
                              random_state=5,max_depth=4,
                              )
model.fit(X_scaled,y_train)
t_scaled=sc.transform(X_test)
p_test=model.predict(t_scaled)
print('accuracy :',accuracy_score(y_test,p_test))


# ## LogisticRegression

# In[ ]:


log_reg = LogisticRegression()
log_reg.fit(X_scaled,y_train)
t_scaled=sc.transform(X_test)
p_test=log_reg.predict(t_scaled)
print('accuracy :',accuracy_score(y_test,p_test))


# In[ ]:


pred=model.predict(test_scaled)
final_df=pd.DataFrame({ 'PassengerId' : test_id, 'Survived': pred })
final_df.to_csv('final_dataset.csv',index=False)


# In[ ]:





# In[ ]:




