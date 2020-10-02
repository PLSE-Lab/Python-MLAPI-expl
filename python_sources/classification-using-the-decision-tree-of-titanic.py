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


# In[ ]:


import seaborn as sns


# In[ ]:


import pandas as pd
df=pd.read_csv("../input/titanic/train.csv")


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df.isnull().sum()


# In[ ]:


test=pd.read_csv("../input/titanic/test.csv")
test.head()


# Let's delete Cabin column and replace the missing Age values with the average age
# 

# In[ ]:


df=df.drop(columns=['Cabin'])


# In[ ]:


df.groupby(['Embarked','Survived'], as_index=False)[['PassengerId']].count()


# In[ ]:


df['Embarked'].fillna('S', inplace=True)


# In[ ]:


sns.countplot(x="Embarked", hue="Survived", data=df)


# In[ ]:


df['Age'].fillna(df['Age'].median(), inplace=True)
df=df.dropna()

df.shape


# In[ ]:


df.corr()


# In[ ]:


df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
df.head()


# df['Title']=df['Title'].map({"Mr":0,"Miss":1,"Mrs":2,"Master":3,"Dr":3,"Rev":3,"Major":3,"Col":3,"Mlle":3,"Don":3,"Jonkheer":3,"Countess":3,"Sir":0,"Capt":3,"Mme":2,"Lady":1,"Ms":0,"Dona":3})
# df.head()

# In[ ]:


df["Age"]=pd.cut(df["Age"],5)


# In[ ]:


df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


df['Alone'] = df['FamilySize'].map(lambda s: 1 if s == 1 else 0)


# Recode columns containing categories
# 

# In[ ]:


from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['Sex'] = LE.fit_transform(df['Sex'])
df['Embarked'] = LE.fit_transform(df['Embarked'])
df['Age'] = LE.fit_transform(df['Age'])
df['Title'] = LE.fit_transform(df['Title'])



# Divide the sample into features and labels

# In[ ]:


features=df[['Pclass','Sex','Age','SibSp','Parch','Embarked','Title','Alone','FamilySize']]


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(features)
features=scaler.transform(features)
features


# In[ ]:


labeles=df["Survived"]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labeles, test_size=0.2,random_state=5 )
from sklearn import tree 
clf=tree.DecisionTreeClassifier(min_samples_split=70)
clf.fit(X_train,y_train)


# In[ ]:


pred=clf.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
s=confusion_matrix(y_test, pred)
s


# In[ ]:


from sklearn.metrics import accuracy_score
accur=accuracy_score(y_test,pred)
print(accur)


# In[ ]:


test_data=pd.read_csv("../input/titanic/test.csv")
test_data.head()


# In[ ]:


test_data['Title'] = test_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test_data.head()


# test_data['Title']=test_data['Title'].map({"Mr":0,"Miss":1,"Mrs":2,"Master":3,"Dr":3,"Rev":3,"Major":3,"Col":3,"Mlle":3,"Don":3,"Jonkheer":3,"Countess":3,"Sir":0,"Capt":3,"Mme":2,"Lady":1,"Ms":0,"Dona":3})
# test_data.head()

# In[ ]:


test=test_data.drop(columns=['PassengerId','Name'])


# In[ ]:


test['FamilySize'] = test['SibSp'] + test['Parch'] + 1


# In[ ]:


test['Alone'] = test['FamilySize'].map(lambda s: 1 if s == 1 else 0)


# In[ ]:


test.isnull().sum()


# In[ ]:


test["Age"].fillna(test['Age'].median(), inplace=True)


# In[ ]:


test["Age"]=pd.cut(test["Age"],5)


# In[ ]:


test=test[['Pclass','Sex','Age','SibSp','Parch','Embarked','Title','Alone','FamilySize']]


# In[ ]:


test.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
test['Sex'] = LE.fit_transform(test['Sex'])
test['Embarked'] = LE.fit_transform(test['Embarked'])
test['Age'] = LE.fit_transform(test['Age'])
test['Title'] = LE.fit_transform(test['Title'])


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(test)
test=scaler.transform(test)
test


# In[ ]:


prediction=clf.predict(test)


# In[ ]:


test_data["Survived"]=prediction


# In[ ]:


submission=test_data[["PassengerId","Survived"]]


# In[ ]:


test_data.head()


# Export the prediction file for the test data

# In[ ]:


submission = pd.DataFrame({ 'PassengerId': test_data.PassengerId.values, 'Survived':test_data.Survived.values  })
submission.to_csv("my_submission_1.csv", index=False)


# In[ ]:


real_data=pd.read_csv("../input/titanic/gender_submission.csv")


# In[ ]:


real_labeles=real_data["Survived"]


#  # Test the accuracy of the model on the new test data

# In[ ]:


from sklearn.metrics import accuracy_score
accur=accuracy_score(real_labeles,prediction)
print(accur)

