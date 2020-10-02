#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# **First let's check our test data**

# In[ ]:


train=pd.read_csv('/kaggle/input/titanic/train.csv')
test=pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()


# **Setting the passender id to our index for easy representation of our data :)**

# In[ ]:


train.set_index('PassengerId')


# 1) Missing data is present or not???

# In[ ]:


train.isnull() # Value which shows True meaning our data is misi=sing or vive-versa


# 1-a) Plotting a heatmap to get the overall look on our missing data :)

# In[ ]:


p=sns.heatmap(train.isnull(),cmap='Blues')
p.set_title('Missing data')


# Our age data is missing a lot and can be filled but Cabin is almost empty so we can drop it for now.
# Now focus on filling the age column and visualising our data

# In[ ]:


Choose_best_variable=train.corr()


# In[ ]:


sns.heatmap(Choose_best_variable,annot=True,cmap='coolwarm')


# > From above we select most correlated data which fits most we will fill it with above 

# In[ ]:


sns.boxplot(x=train['Fare'],y=train['Age']) # As it coorelates most but it didn't seems to be inconsistent


# In[ ]:


sns.boxplot(x=train['SibSp'],y=train['Age']) # Well the data seems to be much more hypothetical


# In[ ]:


sns.boxplot(x=train['Pclass'],y=train['Age']) 


# Above plot give's us much more Accurate result regarding on age

# **Now let's Analyze our data**

# In[ ]:


sns.countplot(x=train['Sex'])


# In[ ]:


sns.countplot(x=train['Survived'],hue=train['Sex'])


# **Well it's been a fact that women lives longer than man and Hence proved :)**

# In[ ]:


sns.countplot(x=train['Survived'],hue=train['Pclass'])


# Person who paid more (Pclass=1) survived more than Pclass(1&2)

# In[ ]:


sns.countplot(x=train['Pclass'])
# Plot showing how many people belonging to different class


# In[ ]:


plt.figure(figsize=(12,4))
sns.kdeplot(train['Fare'])


# Above plot show max people prefer to spend less money on ticket between [0,100]

# In[ ]:


sns.countplot(x=train['SibSp'])


# In[ ]:


sns.countplot(x=train['SibSp'],hue=train['Survived'])


# Well above data show Single people survived most

# **Training our model**

# Combining both data to get our result more accurate

# In[ ]:


data=pd.concat([train,test])
data


# Now look at the missing data 

# In[ ]:


sns.heatmap(data.isnull(),cmap='Blues')


# Well from above we can conclude the following points
# * Age column is missing which can be recovered by boxplot of age vs Pclass
# * Cabin data is missing in a large amount and cannot be recovered
# * The survived which is not showing is the value which is going to be predicted by us

# In[ ]:


def fill(x):
    age=x[0]
    clas=x[1]
    if pd.isnull(age):
        if clas==1:
            return round(data[data['Pclass']==1]['Age'].mean())
        elif clas==2:
            return round(data[data['Pclass']==2]['Age'].mean())
        else:
            return round(data[data['Pclass']==3]['Age'].mean())
    else:
        return age
data['Age']=data[['Age','Pclass']].apply(fill,axis=1)
data.head()


# In[ ]:


sns.heatmap(data.isnull(),cmap='Blues')


# Well data can take only Categorical values only so let's convert it into required form

# In[ ]:


a=pd.get_dummies(data['Sex'],drop_first=True)
a # Converting it into 0 & 1 form to interpret easily


# In[ ]:


b=pd.get_dummies(data['Embarked'],drop_first=True)
b


# In[ ]:


data=pd.concat([data,a,b],axis=1)
data.head(1)


# Now removing the waste things which need not to be included in trainig our model

# In[ ]:


data.drop(['PassengerId','Name','Sex','Ticket','Cabin','Embarked'],axis=1,inplace=True)
data.head()


# As of now we are having clean data so we can procced it to training part of our model

# In[ ]:


# Reconverting the data
train=data[:len(train)]
test=data[:len(test)]
sns.heatmap(train.isnull(),cmap='Blues')


# Above graph show that we have our complete data

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


y=train['Survived']
x=train.drop('Survived',axis=1)


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=101,test_size=0.3)


# We will be testing different methods and selecting the best among them 

# In[ ]:


cross_val_score(RandomForestClassifier(),x,y).mean()


# In[ ]:


cross_val_score(LogisticRegression(),x,y).mean()


# Hence from the above 2 we will use the Random Forest Classifier

# In[ ]:


trainmodel=LogisticRegression()


# In[ ]:


trainmodel.fit(x_train,y_train)


# In[ ]:


data_to_test=test.drop('Survived',axis=1)


# In[ ]:


result=trainmodel.predict(data_to_test)


# In[ ]:


result


# In[ ]:


test_again=pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


work=pd.DataFrame({'PassengerId': test_again['PassengerId'], 'Survived':result})
final_work=work.round(0).astype(int)


# In[ ]:


final_work.to_csv('Result.csv',index = False)
final_work


# In[ ]:


test


# In[ ]:


sns.heatmap(test.isnull(),cmap='Blues')


# In[ ]:




