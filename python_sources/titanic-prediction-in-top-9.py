#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('train.csv')


# In[ ]:


df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:





# In[ ]:


df.head()


# In[ ]:


df.isnull().sum()


# In[ ]:


df['Age'].fillna(df['Age'].median() ,inplace = True)


# In[ ]:


df.isnull().sum()


# In[ ]:


df['Embarked'].fillna('S' , inplace = True)


# In[ ]:


df.isnull().sum()


# In[ ]:


df['Title'].value_counts()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.info()


# In[ ]:





# In[ ]:


df.head()


# In[ ]:


df['Cabin'] = df['Cabin'].fillna('Unknown')


# In[ ]:


df['Cabin'] = df['Cabin'].str[0]


# In[ ]:


df.head(5)


# In[ ]:





# In[ ]:


df['Cabin'] = np.where((df.Pclass==1) & (df.Cabin=='U'),'C',
                        np.where((df.Pclass==2) & (df.Cabin=='U'),'D',
                        np.where((df.Pclass==3) & (df.Cabin=='U'),'G',
                        np.where(df.Cabin=='T','C',df.Cabin))))


# In[ ]:


df.head()


# In[ ]:


df['Cabin'].value_counts()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Cabin'] = le.fit_transform(df['Cabin'])


# In[ ]:


df.head()


# In[ ]:





# In[ ]:


df = df.drop('Ticket' , axis = 1)


# In[ ]:


df = df.drop('Name' , axis = 1)


# In[ ]:


df = df.drop('PassengerId' , axis = 1)


# In[ ]:


df.head()


# In[ ]:


le = LabelEncoder()
df['Title'] = le.fit_transform(df['Title'])


# In[ ]:


le = LabelEncoder()
df['Embarked'] = le.fit_transform(df['Embarked'])


# In[ ]:


le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])


# In[ ]:


df.head()


# In[ ]:





# In[ ]:


df.head()


# In[ ]:


df['fam_size'] = df['SibSp'] + df['Parch'] + 1


# In[ ]:


df.head()


# In[ ]:





# In[ ]:


X_train = df.drop("Survived" , axis =1)
Y_train = df['Survived']


# In[ ]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(X_train)
X_train = ss.transform(X_train)


# In[ ]:


df.isnull().sum()


# In[ ]:





# In[ ]:





# In[ ]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth= 8)
model.fit(X_train , Y_train)


# In[ ]:





# In[ ]:


t = pd.read_csv('test.csv')


# In[ ]:





# In[ ]:


t.head()


# In[ ]:


t.isnull().sum()


# In[ ]:


t['Title'] = t.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


t.head()


# In[ ]:


t['Title'] = np.where((t.Title=='Capt') | (t.Title=='Countess') | (t.Title=='Don') | (t.Title=='Dona')
                        | (t.Title=='Jonkheer') | (t.Title=='Lady') | (t.Title=='Sir') | (t.Title=='Major') | (t.Title=='Rev') | (t.Title=='Col'),'Other',t.Title)

t['Title'] = t['Title'].replace('Ms','Miss')
t['Title'] = t['Title'].replace('Mlle','Miss')
t['Title'] = t['Title'].replace('Mme','Mrs')


# In[ ]:


t.head()


# In[ ]:


t['Age'] = np.where((t.Age.isnull()) & (t.Title=='Master'),5,
                        np.where((t.Age.isnull()) & (t.Title=='Miss'),22,
                                 np.where((t.Age.isnull()) & (t.Title=='Mr'),32,
                                          np.where((t.Age.isnull()) & (t.Title=='Mrs'),37,
                                                  np.where((t.Age.isnull()) & (t.Title=='Other'),45,
                                                           np.where((t.Age.isnull()) & (t.Title=='Dr'),44,t.Age))))))  


# In[ ]:


t.isnull().sum()


# In[ ]:


t.info()


# In[ ]:


t['Fare'] = t['Fare'].fillna(t['Fare'].mean())


# In[ ]:


t['Cabin'] = t['Cabin'].fillna('Unknown')
t['Cabin'] = t['Cabin'].str[0]


# In[ ]:


t['Cabin'] = np.where((t.Pclass==1) & (t.Cabin=='U'),'C',
                        np.where((t.Pclass==2) & (t.Cabin=='U'),'D',
                        np.where((t.Pclass==3) & (t.Cabin=='U'),'G',
                        np.where(t.Cabin=='T','C',t.Cabin))))


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
t['Cabin'] = le.fit_transform(t['Cabin'])


# In[ ]:


le = LabelEncoder()
t['Embarked'] = le.fit_transform(t['Embarked'])


# In[ ]:


le = LabelEncoder()
t['Sex'] = le.fit_transform(t['Sex'])


# In[ ]:


le = LabelEncoder()
t['Title'] = le.fit_transform(t['Title'])


# In[ ]:


t['fam_size'] = t['Parch'] + t['SibSp'] + 1


# In[ ]:


t.isnull().sum()


# In[ ]:


t = t.drop('PassengerId' ,axis =1)
t = t.drop('Name' ,axis =1)
t = t.drop('Ticket' ,axis =1)


# In[ ]:


t.head()


# In[ ]:



from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(t)
t = ss.transform(t)


# In[ ]:





# In[ ]:


t[0]


# In[ ]:





# In[ ]:


pred = model.predict(t)


# In[ ]:


pd.DataFrame(pred).to_csv('file.csv')


# In[ ]:




