#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))


# In[2]:


df=pd.read_csv('../input/train.csv')


# In[3]:


df.head()


# In[4]:


df['Embarked'] = df['Embarked'].replace('C','Cherbourg')
df['Embarked'] = df['Embarked'].replace('S','Southampton')
df['Embarked'] = df['Embarked'].replace('Q','Queenstown')


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df.isnull().any()


# In[8]:


sns.countplot(df['Embarked'])


# In[9]:


df['Embarked'] = df['Embarked'].fillna('Southampton')


# In[10]:


Sex = pd.get_dummies(df.Sex)
Sex.drop(Sex.columns[1],axis=1,inplace=True)
df = pd.concat([df,Sex],axis = 1)

Embarked = pd.get_dummies(df.Embarked)
Embarked.drop(Embarked.columns[1],axis=1,inplace=True)
df = pd.concat([df,Embarked],axis = 1)


# In[11]:


df.describe()


# In[12]:


sns.countplot(x='Survived',hue='Sex',data=df)


# not many survived

#  out of survived more are females

# In[13]:


sns.barplot(x='Survived',y='Age',hue='Sex',data=df)


# In[14]:


def bar_chart(feature):
   survived = df[df['Survived']==1][feature].value_counts()
   dead = df[df['Survived']==0][feature].value_counts()
   df1 = pd.DataFrame([survived,dead])
   df1.index = ['Survived','Dead']
   df1.plot(kind='bar',stacked=True, figsize=(5,5))


#  there are more singles on the board
#  
#  out of the survived singles are most but contrary is singles are most in the dead too

# In[15]:


sns.countplot(x=df['Survived'],hue=df['SibSp'])


# there are more singles on the board
# 
# out of the survived singles are most but contrary is singles are most in the dead too

# In[16]:


sns.countplot(x=df['Survived'],hue=df['Parch'])


#  out of survided 1st class are more
# 
# 3rd class died the most
# 
# there are almost equal deaths and survivals in 2nd class

# In[17]:


sns.countplot(x=df['Survived'],hue=df['Pclass'])


#  there more people from southampton
#  
#  proportion of survived from Cherbourg are most.
#  
#  3/4 th of southamptam had no good fate

# In[18]:


sns.countplot(x=df['Survived'],hue=df['Embarked'])


# In[19]:


sns.catplot(data = df, x ="Survived", y = "Age", 
               col = 'Pclass', # per store type in cols
               hue = 'female',
              sharex=False)


#   person of age above 70 survided only from class 1

#  death rate is very high in class 3
# 
# most of the females from class 1 and 2 survived

# In[20]:


print('Class 1 females')
print('total',df[(df['Pclass']==1) & (df['female']==1)]['Survived'].count())
print('survived',df[(df['Pclass']==1) & (df['female']==1)]['Survived'].sum())

print('\n')
print('Class 2 females')
print('total',df[(df['Pclass']==2) & (df['female']==1)]['Survived'].count())
print('survived',df[(df['Pclass']==2) & (df['female']==1)]['Survived'].sum())

print('\n')
print('Class 3 females')
print('total',df[(df['Pclass']==3) & (df['female']==1)]['Survived'].count())
print('survived',df[(df['Pclass']==3) & (df['female']==1)]['Survived'].sum())


# In[21]:


print('Class 1 children')
print('total',df[(df['Pclass']==1) & (df['Age']<=10)]['Survived'].count())
print('survived',df[(df['Pclass']==1) & (df['Age']<=10)]['Survived'].sum())
print('\n')
print('Class 2 children')
print('total',df[(df['Pclass']==2) & (df['Age']<=10)]['Survived'].count())
print('survived',df[(df['Pclass']==2) & (df['Age']<=10)]['Survived'].sum())
print('\n')
print('Class 3 children')
print('total',df[(df['Pclass']==3) & (df['Age']<=10)]['Survived'].count())

print('survived',df[(df['Pclass']==3) & (df['Age']<=10)]['Survived'].sum())


# In[22]:


print('Cherbourg females')
print('total',df[(df['Embarked']=='Cherbourg') & (df['female']==1)]['Survived'].count())
print('survived',df[(df['Embarked']=='Cherbourg') & (df['female']==1)]['Survived'].sum())

print('\n')
print('Southampton females')
print('total',df[(df['Embarked']=='Southampton') & (df['female']==1)]['Survived'].count())
print('survived',df[(df['Embarked']=='Southampton') & (df['female']==1)]['Survived'].sum())

print('\n')
print('Queenstown females')
print('total',df[(df['Embarked']=='Queenstown') & (df['female']==1)]['Survived'].count())
print('survived',df[(df['Embarked']=='Queenstown') & (df['female']==1)]['Survived'].sum())


# In[23]:


emdf =df
emdf['Embarked'] = emdf['Embarked'].replace('Cherbourg',0)
emdf['Embarked'] = emdf['Embarked'].replace('Southampton',1)
emdf['Embarked'] = emdf['Embarked'].replace('Queenstown',2)


# In[24]:


sns.catplot(data = df, x ="Survived", y = "Age", 
               col = 'Pclass', # per store type in cols
               hue = 'female',
               row = 'Embarked',
               sharex=False)


#  age of people is almost a normal distribution, so we can fill the nan with mean

# In[25]:


sns.distplot(df['Age'].dropna())


# In[26]:


df['Age'] = df['Age'].fillna(np.mean(df['Age']))


# In[27]:


sns.distplot(df['Age'].dropna())


# In[28]:


sns.catplot(data = df, x ="Survived", y = "Age", sharex=False)


#  as the person grows old survival chances are less

# In[29]:


sns.catplot(data = df[(df['female']==1) | (df['Age']<=10)], x ="Survived", y = "Age", sharex=False)


#  Most of the children and females survived

# 
# fare is not a normal distribution as we already know that there are more 3rd class on the ship

# In[30]:


sns.distplot(df['Fare'])


# In[31]:


df['Fare'] = df['Fare'].replace(0,np.median(df['Fare']))


# In[32]:


plt.hist(df['Fare'])


# In[33]:


sns.countplot(x=df['Fare'],hue=df['Pclass'])


# In[34]:


sns.distplot(np.log(df['Fare']))


# In[35]:


ndf = df
ndf.info()


# In[36]:


ndf.describe()


# In[37]:


pd.crosstab([df.Embarked,df.Pclass],[df.Sex,df.Survived],margins=True).style.background_gradient(cmap='Blues')


# In[38]:


plt.figure(figsize=(10,7))
sns.heatmap(ndf.corr(),cmap='Blues',annot=True)


# In[39]:


tedf = pd.read_csv('../input/test.csv')
tedf.isnull().any()


# In[40]:


tedf.info()


# In[41]:


tedf['Embarked'] = tedf['Embarked'].replace('C','Cherbourg')
tedf['Embarked'] = tedf['Embarked'].replace('S','Southampton')
tedf['Embarked'] = tedf['Embarked'].replace('Q','Queenstown')


# In[42]:


Sex = pd.get_dummies(tedf.Sex)
Sex.drop(Sex.columns[1],axis=1,inplace=True)
tedf = pd.concat([tedf,Sex],axis = 1)

Embarked = pd.get_dummies(tedf.Embarked)
Embarked.drop(Embarked.columns[1],axis=1,inplace=True)
tedf = pd.concat([tedf,Embarked],axis = 1)


# In[43]:


tedf.isnull().any()


# In[44]:


features = ['Pclass','Age','Fare','female','Cherbourg','Southampton']


# In[45]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[46]:


def conflog(features):
    X = ndf[features]
    y = ndf.Survived
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=125)
    linreg = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
    linreg.fit(X_train,y_train)
    y_pred = linreg.predict(X_test)   
    print(metrics.classification_report(y_test,y_pred))
    print(metrics.accuracy_score(y_test,y_pred))


# In[47]:


print(conflog(features))


# In[48]:


print(conflog(['Pclass','female','Cherbourg','Southampton']))


# In[49]:


from sklearn.naive_bayes import GaussianNB


# In[50]:


def confNB(features):
    X = ndf[features]
    y = ndf.Survived
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=125)
    nb = GaussianNB()
    nb.fit(X_train,y_train)
    y_pred = nb.predict(X_test)   
    print(metrics.classification_report(y_test,y_pred))
    print(metrics.accuracy_score(y_test,y_pred))


# In[51]:


print(confNB(features))


# In[52]:


print(confNB(['Pclass','female','Cherbourg','Southampton']))


# In[53]:


from sklearn.tree import DecisionTreeClassifier


# In[54]:


def confDT(features):
    X = ndf[features]
    y = ndf.Survived
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=125)
    dt = DecisionTreeClassifier()
    dt.fit(X_train,y_train)
    y_pred = dt.predict(X_test)   
    print(metrics.classification_report(y_test,y_pred))
    print(metrics.accuracy_score(y_test,y_pred))


# In[55]:


print(confDT(features))


# In[56]:


print(confDT(['Pclass','female','Cherbourg','Southampton']))


# In[57]:


from sklearn.svm import SVC


# In[58]:


def confsvm(features):
    X = ndf[features]
    y = ndf.Survived
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=125)
    svmf = SVC(gamma='scale')
    svmf.fit(X_train,y_train)
    y_pred = svmf.predict(X_test)   
    print(metrics.classification_report(y_test,y_pred))
    print(metrics.accuracy_score(y_test,y_pred))


# In[59]:


print(confsvm(features))


# In[60]:


print(confsvm(['Pclass','female','Cherbourg','Southampton']))


# In[61]:


X_train = ndf[['Pclass','female','Cherbourg','Southampton']]
y_train = ndf.Survived
X_test = tedf[['Pclass','female','Cherbourg','Southampton']]
svmf = SVC(gamma='scale')
svmf.fit(X_train,y_train)
y_pred = svmf.predict(X_test)


# In[62]:


y_pred

