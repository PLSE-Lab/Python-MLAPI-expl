#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# In[ ]:


train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


train.head(4)


# In[ ]:


train.info()


# In[ ]:


sns.heatmap(train.isnull(), cbar=False, yticklabels=False, cmap='viridis')


# In[ ]:


sns.heatmap(test.isnull(), cbar=False, yticklabels=False, cmap='viridis')


# In[ ]:


sns.set_style('whitegrid')


# In[ ]:


plt.figure(figsize=(8,6))
p1 = sns.countplot(data = train, x = 'Survived', hue = 'Sex', palette='coolwarm')
p1.set_title('Comparing Survival rate based on gender',pad=30,fontsize=20)
p1.set_xlabel('')
p1.set_xticklabels(['Not Survive','Survive'],fontsize=15)


# In[ ]:


#To understand classwise survival rate
sns.countplot(data = train, x = 'Survived', hue = 'Pclass')
#Class 3(cheapest) survival was least


# In[ ]:


plt.figure(figsize=(11,7))
sns.distplot(train['Age'].dropna(), bins = 30, kde=False)
#graph skewed towards kids. More people of age group 20-30


# In[ ]:


plt.figure(figsize=(8,5))
sns.countplot(x='SibSp', data=train)
#Most people on board were travelling alone. Some were either with spouse or single parent


# In[ ]:


train.info()


# In[ ]:


import cufflinks as cf
cf.go_offline()


# In[ ]:


train['Fare'].iplot(kind='hist',bins=80, yTitle='Fare', xTitle='No. of Tickets')
#most people on board 3rd class = cheap ticket


# In[ ]:


#Exploring Age
plt.figure(figsize=(9,7))
sns.boxplot(x='Pclass', y='Age',data=train)


# In[ ]:


mean_ages = train.groupby('Pclass').mean()['Age']
mean_ages


# In[ ]:


def fill_age(col):
    age = col[0]
    pclass = col[1]
    
    if pd.isnull(age):
        return mean_ages[pclass]
    else:
        return age


# In[ ]:


#fill_age([None,1])


# In[ ]:


train['Age']=train[['Age','Pclass']].apply(fill_age,axis=1)
test['Age']=test[['Age','Pclass']].apply(fill_age,axis=1)


# In[ ]:


sns.heatmap(train.isnull(), cbar=False, yticklabels=False, cmap='viridis')


# In[ ]:


sns.heatmap(test.isnull(), cbar=False, yticklabels=False, cmap='viridis')


# In[ ]:


train.drop('Cabin',axis=1,inplace = True)
test.drop('Cabin',axis=1,inplace = True)


# In[ ]:


train.fillna('S',inplace=True)


# In[ ]:


test['Fare'].fillna(test['Fare'].mean(),inplace=True)


# In[ ]:


sns.heatmap(train.isnull(), cbar=False, yticklabels=False, cmap='viridis')


# In[ ]:


sns.heatmap(test.isnull(), cbar=False, yticklabels=False, cmap='viridis')


# ## Feature Engineering of Categorical Data - Dummy Var

# In[ ]:


train.head(5)


# In[ ]:


males = pd.get_dummies(train['Sex'],drop_first=True)
emb = pd.get_dummies(train['Embarked'],drop_first=True)
pclss = pd.get_dummies(train['Pclass'],drop_first=True)

males2 = pd.get_dummies(test['Sex'],drop_first=True)
emb2 = pd.get_dummies(test['Embarked'],drop_first=True)
pclss2 = pd.get_dummies(test['Pclass'],drop_first=True)


# In[ ]:


train = pd.concat([train,emb,males,pclss],axis=1)
test = pd.concat([test,emb2,males2,pclss2],axis=1)


# In[ ]:


train.rename(columns = {2:'Pclass 2', 3:'Pclass 3'},inplace=True) 
test.rename(columns = {2:'Pclass 2', 3:'Pclass 3'},inplace=True) 


# In[ ]:


train.head(2)


# In[ ]:


test.head(2)


# In[ ]:


train.drop(['Name','Ticket','Embarked','Sex','Pclass'],axis=1,inplace=True)
test.drop(['Name','Ticket','Embarked','Sex','Pclass'],axis=1,inplace=True)


# In[ ]:


train.head(4)


# In[ ]:


test.head(4)


# In[ ]:


train.tail(3)


# In[ ]:


train.shape


# In[ ]:


TestPass_id = test['PassengerId']
train.drop('PassengerId',axis=1,inplace = True)
test.drop('PassengerId',axis=1,inplace = True)


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


std = StandardScaler()


# In[ ]:


train[['Age','Fare']] = std.fit_transform(train[['Age','Fare']])
test[['Age','Fare']] = std.fit_transform(test[['Age','Fare']])


# In[ ]:


train.head(5)


# In[ ]:


test.head(5)


# # Training Model
# 

# In[ ]:


x = train.drop('Survived',axis=1)
y = train['Survived']


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(criterion='gini', n_estimators=700,)


# In[ ]:


rf.fit(x,y)


# In[ ]:


pred = rf.predict(test)


# In[ ]:


submission = pd.DataFrame({'PassengerId':TestPass_id,'Survived':pred})


# In[ ]:


submission


# In[ ]:


filename = 'Titanic Predictions 4.csv'
submission.to_csv(filename,index=False)

print('Saved file: ' + filename)


# In[ ]:


#76% accuracy


# In[ ]:




