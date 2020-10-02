#!/usr/bin/env python
# coding: utf-8

# This is my first Machine learning model. I've used logistic regression which is giving 75% score and random forest which gave me 77% score. 
# I have used Age, Pclass, Cabin, Fare, Sex and Embarked variables. Now I'm stuck and unable to improve the score. **Welcome all suggestions**. 

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


# In[ ]:


# importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# importing the datasets 

dataset_train = pd.read_csv("../input/titanic/train.csv")
dataset_test = pd.read_csv("../input/titanic/test.csv")
dataset_train.head()


# In[ ]:


# dropping the name column
dataset_train.drop(["Name"], axis=1, inplace= True)
dataset_test.drop(["Name"], axis=1, inplace= True)
dataset_train['related'] = dataset_train['SibSp'] + dataset_train['Parch']
dataset_test['related'] = dataset_test['SibSp'] + dataset_test['Parch']
dataset_train.head()


# In[ ]:


# visualizing the data to know the relation between survival and other parameters
sns.barplot(x='Sex',y='Survived',data=dataset_train)


# In[ ]:


# high survival rate if you boarded from Cherbyl
sns.barplot(x='Embarked',y='Survived',data=dataset_train)


# In[ ]:


sns.barplot(x='Pclass',y='Survived',data=dataset_train)


# In[ ]:


sns.barplot(x='related',y='Survived',data=dataset_train)


# In[ ]:


# handling NAN entries in the dataset
# for the Age column, I have first grouped the data according to 
# sex and Class and then taken the median
dataset_train.isnull().sum()
grouped = dataset_train.groupby(['Sex','Pclass'])  
grouped.Age.median()
dataset_train.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))
grouped = dataset_test.groupby(['Sex','Pclass']) 
dataset_test.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))

dataset_train['Fare']=dataset_train['Fare'].fillna(dataset_train['Fare'].median())
dataset_test['Fare']=dataset_test['Fare'].fillna(dataset_test['Fare'].median())
dataset_train['Cabin'] = dataset_train['Cabin'].fillna('U')
dataset_test['Cabin'] = dataset_test['Cabin'].fillna('U')
dataset_train['Cabin'] = [x[:1] for x in dataset_train['Cabin']]
dataset_test['Cabin'] = [x[:1] for x in dataset_test['Cabin']]

dataset_train['Embarked'].describe()
dataset_train['Embarked']=dataset_train['Embarked'].fillna('S')
dataset_test['Embarked']=dataset_test['Embarked'].fillna('S')
dataset_train.isnull().sum()


# In[ ]:


# handling Categorical values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
dataset_train['Sex']= labelencoder.fit_transform(dataset_train['Sex'])
dataset_test['Sex']= labelencoder.fit_transform(dataset_test['Sex'])
#dataset_train['Sex'].unique() 
onehotencoder = OneHotEncoder(handle_unknown='ignore')
dataset_train=dataset_train.join(pd.DataFrame(onehotencoder.fit_transform(dataset_train[['Embarked']]).toarray()))
dataset_test = dataset_test.join(pd.DataFrame(onehotencoder.fit_transform(dataset_test[['Embarked']]).toarray()))
cabin_dummies = pd.get_dummies(dataset_train.Cabin, prefix="Cabin")
dataset_train = pd.concat([dataset_train,cabin_dummies],axis=1)
cabin_dummies = pd.get_dummies(dataset_test.Cabin, prefix="Cabin")
dataset_test = pd.concat([dataset_test,cabin_dummies],axis=1)

class_dummies = pd.get_dummies(dataset_train.Pclass, prefix="Pclass")
dataset_train = pd.concat([dataset_train,class_dummies],axis=1)
class_dummies = pd.get_dummies(dataset_test.Pclass, prefix="Pclass")
dataset_test = pd.concat([dataset_test,class_dummies],axis=1)
dataset_train.head()


# In[ ]:



X = dataset_train.loc[:,["Age","Fare","Sex","related","Cabin_A","Cabin_B","Cabin_C","Cabin_D","Cabin_E","Cabin_F","Cabin_G","Cabin_T","Cabin_U",0,1,2,"Pclass_1","Pclass_2","Pclass_3"]]
Y = dataset_train.loc[:,["Survived"]]
print(X)
print(Y)
dataset_test["Cabin_T"]=0


# In[ ]:



from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2) 
test = dataset_test.loc[:,["Age","Fare","Sex","related","Cabin_A","Cabin_B","Cabin_C","Cabin_D","Cabin_E","Cabin_F","Cabin_G","Cabin_T","Cabin_U",0,1,2,"Pclass_1","Pclass_2","Pclass_3"]]

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
logisticregressor = LogisticRegression()
logisticregressor.fit(X_test,Y_test)
p1 = logisticregressor.predict(test)
output = pd.DataFrame({"PassengerId" : dataset_test.PassengerId, "Survived" : p1})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
acc1 = logisticregressor.score(X_test,Y_test)
print(acc1)


# In[ ]:



random = RandomForestClassifier(n_estimators=10,random_state=0)
random.fit(X,Y)
p2 = random.predict(test)
output2 = pd.DataFrame({"PassengerId" : dataset_test.PassengerId, "Survived" : p2})
output2.to_csv('submission_1.csv', index=False)
print("Your submission was successfully saved!")

acc2 = random.score(X,Y)
print(acc2)

