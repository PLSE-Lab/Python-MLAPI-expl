#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


test_file= '/kaggle/input/titanic/test.csv'
train_file= '/kaggle/input/titanic/train.csv'


# In[ ]:


train_df=pd.read_csv(train_file)
test_df=pd.read_csv(test_file)


# In[ ]:


#train_df.head()
#test_df.head()


# In[ ]:


train_df.shape


# In[ ]:


test_df.info()


# In[ ]:


test_df.isnull().sum()


# In[ ]:


import matplotlib.pyplot as plot
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()


# In[ ]:


def barchart(feature):
    survived=train_df[train_df['Survived']==1][feature].value_counts()
    dead=train_df[train_df['Survived']==0][feature].value_counts()
    df=pd.DataFrame([survived,dead])
    df.index=['Survived','Dead']
    df.plot(kind='bar', stacked=True,figsize=(16,10))
    
    
    


# In[ ]:


barchart('Sex')


# In[ ]:


barchart('Pclass')


# In[ ]:


barchart('SibSp')


# In[ ]:


train_test_data=[train_df,test_df]

for dataset in train_test_data:
    dataset['Title']=dataset['Name'].str.extract('([A-Za-z]+)\.',expand=False) #letters followed by a dot

train_df.head()


# In[ ]:


train_df['Title'].value_counts()


# In[ ]:


title_mapping={"Mr":0,"Miss":1,"Mrs":2,
               "Master":3,"Dr":3,"Rev":3,"Major":3,"Col":3,"Mlle":3,"Countess":3,"Mme":3,"Capt":3,"Ms":3,"Don":3,"Jonkheer":3,"Lady":3}
for dataset in train_test_data:
    dataset['Title']=dataset['Title'].map(title_mapping)
    
train_df.head()


# In[ ]:


barchart('Title')


# In[ ]:


train_df.drop('Name',axis=1, inplace=True)
#train_df.head()
test_df.drop('Name',axis=1, inplace=True)


# In[ ]:


test_df.head()


# In[ ]:


#train_test_data=[train_df,test_df]
sex_mapping={"male":0,"female":1}
for dataset in train_test_data:
    dataset['Sex']= dataset['Sex'].map(sex_mapping)   


# In[ ]:


train_df.head()


# In[ ]:


barchart('Sex')


# ### Missing Age

# In[ ]:


train_df['Age'].fillna(train_df.groupby('Title')['Age'].transform('median'),inplace=True)
test_df['Age'].fillna(test_df.groupby('Title')['Age'].transform('median'),inplace=True)


# In[ ]:


sns.set_style("darkgrid")

sns.kdeplot(data=train_df[train_df['Survived']==0]['Age'],label='Mr',shade=True)
sns.kdeplot(data=train_df[train_df['Survived']==1]['Age'],label='Mrs',shade=True)


# In[ ]:


#facet=sns.FacetGrid(train_df,hue='Survived',aspect=3)
#facet.map(sns.kdeplot,'Age',shade=True)
#facet.set(xlim=(0,train_df['Age'].max()))
#plt.legend()

#plt.show()


# In[ ]:


train_df.head()


# In[ ]:


train_df.drop('Ticket',axis=1, inplace=True)
train_df.drop('Cabin',axis=1, inplace=True)
train_df.drop('Embarked',axis=1, inplace=True)
target = train_df['Survived']


# In[ ]:


train_df.head()

train_df.loc[599,'Title']=1.0


# In[ ]:


train_df['Title'].isnull().sum()


# In[ ]:


train_df.head()


# In[ ]:


from sklearn.svm import SVC


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold=KFold(n_splits=10,shuffle=True,random_state=0)


# In[ ]:


clf=SVC()
scoring='accuracy'
#help(cross_val_score)
score=cross_val_score(clf,train_df,target,cv=k_fold,n_jobs=1,scoring=scoring)
print(score)


# In[ ]:


round(np.mean(score)*100,3)


# In[ ]:


test_df.head()


# In[ ]:


test_df.drop('Ticket',axis=1, inplace=True)
test_df.drop('Cabin',axis=1, inplace=True)
test_df.drop('Embarked',axis=1, inplace=True)


# In[ ]:


#test_df[test_df['Fare'].isnull()]
test_df.loc[152,'Fare']=12.20


# In[ ]:


#test_df[test_df['Title'].isnull()]
test_df.loc[414,'Title']=1.0


# In[ ]:


train_df.drop('Survived',axis=1, inplace=True)


# In[ ]:


train_df.head()


# In[ ]:


clf.fit(train_df,target)
prediction=clf.predict(test_df)


# In[ ]:


submission=pd.DataFrame({"PassengerId":test_df['PassengerId'],"Survived":prediction})


# In[ ]:


submission.to_csv('submission.csv')


# In[ ]:


submission.head()


# In[ ]:




