#!/usr/bin/env python
# coding: utf-8

# **Titanic: Machine Learning from Disaster**
# 
# ***    ACCURACY: OVER 80% ***
# ***    Kaggle top 8%***
# 
# Github: https://github.com/garvitkhurana/Titanic_ML_disaster
# 
# LinkedIN:  https://www.linkedin.com/in/garvitkhurana/
# 
# Comment if any errors or doubts
# 

# In[12]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,StratifiedKFold,cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler


# In[14]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[15]:


train.head()


# In[16]:


train.info()


# In[17]:


train.describe()


# In[18]:


train.hist(figsize=(20,12))


# In[19]:


def  bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))
    


# In[20]:


bar_chart("Sex")


# In[21]:


bar_chart("Pclass")


# In[22]:


bar_chart("SibSp")


# In[23]:


bar_chart('Embarked')


# In[24]:


sns.barplot(y="Survived",x="Sex",data=train)


# In[25]:


sns.barplot(y="Survived",x="Embarked",data=train)


# In[26]:


t=sns.barplot(y="Survived",x="Parch",data=train)


# In[27]:


t=sns.barplot(y="Survived",x="SibSp",data=train)


# In[32]:


train = pd.read_csv("../input/train.csv")
y=train.Survived.values
test = pd.read_csv("../input/test.csv")
pas_id=test.PassengerId.values
train_test_data = pd.concat([train, test],axis=0)


# In[33]:


train_test_data.head()


# In[34]:


train_test_data.shape


# In[35]:


title=[]
for i in train_test_data['Name']:
    t=i.split(",")[1].split(".")[0].strip()
    title.append(t)
train_test_data["Title"]=title


# In[36]:


train_test_data.Title.value_counts()


# In[37]:


train_test_data['FamilySize'] = train_test_data['Parch'] + train_test_data['SibSp'] + 1


# In[38]:


train_test_data.FamilySize.value_counts()


# In[39]:


train_test_data['Singleton'] = train_test_data['FamilySize'].map(lambda s: 1 if s == 1 else 0)
train_test_data['SmallFamily'] = train_test_data['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
train_test_data['LargeFamily'] = train_test_data['FamilySize'].map(lambda s: 1 if 5 <= s else 0)


# In[40]:


CleanTicket=[]
for i in train_test_data.Ticket:
    i = i.replace('.', '')
    i = i.replace('/', '')
    i = i.split()[0]
    if i.isalpha():
        CleanTicket.append(i)
    else:
        CleanTicket.append("Number")


# In[41]:


train_test_data["CleanTicket"]=CleanTicket


# In[42]:


plt.figure(figsize=(20,12))
sns.barplot("FamilySize","Survived",data=train_test_data)


# In[43]:


plt.figure(figsize=(20,12))
sns.barplot("Title","Survived",data=train_test_data)


# In[44]:


plt.figure(figsize=(20,12))
sns.barplot("CleanTicket","Survived",data=train_test_data,ci=False)


# In[45]:


train_test_data.drop(["Survived"],axis=1,inplace=True)


# In[46]:


train_test_data.drop(["Name","FamilySize","Ticket"],axis=1,inplace=True)


# In[47]:


def cable_name(x):
    try:
        return x[0]
    except TypeError:
        return "None"
train_test_data["Cabin"]=train_test_data.Cabin.apply(cable_name)


# In[48]:


train_test_data.describe()


# In[49]:


train_test_data['Age'].fillna(np.mean(train_test_data.Age),inplace=True)
train_test_data['Fare'].fillna(np.mean(train_test_data.Fare),inplace=True)
train_test_data['Fare'] = StandardScaler().fit_transform(train_test_data['Fare'].values.reshape(-1, 1))
train_test_data.describe()


# In[50]:


num_data=train_test_data.select_dtypes(exclude=object).columns


# In[51]:


train_test_data[num_data].head()


# In[52]:


cat_data=train_test_data.select_dtypes(include=object).columns


# In[53]:


print(train_test_data[cat_data].info())
train_test_data['Embarked'].fillna(train_test_data['Embarked'].mode()[0], inplace = True)


# In[54]:


for i in cat_data:
    train_test_data[i].fillna("Missing",inplace=True)
    dummies=pd.get_dummies(train_test_data[i],prefix=i)
    train_test_data=pd.concat([train_test_data,dummies],axis=1)
    train_test_data.drop(i,axis=1,inplace=True)


# In[55]:


train_test_data.head()


# In[56]:


test_data=train_test_data.iloc[891:]
train_data=train_test_data.iloc[:891]
y=y
train=train_data
targets=y


# In[57]:


clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(train, targets)


# In[58]:


features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)


# In[59]:


features.plot(kind='barh', figsize=(25, 25))


# In[60]:


model = SelectFromModel(clf, prefit=True)
train_reduced = model.transform(train_data)
print(train_reduced.shape)
test_reduced = model.transform(test_data)
print(test_reduced.shape)


# In[61]:


clf=RandomForestClassifier(min_samples_split=2,max_depth=6,bootstrap=True,min_samples_leaf=1,n_estimators=100,max_features='auto')
clf.fit(train_reduced,y)
pred=clf.predict(test_reduced)


# In[70]:


from sklearn.model_selection import cross_val_score,train_test_split
X_train,X_test,y_train,y_test=train_test_split(train_data,y,test_size=0.2)
clf=RandomForestClassifier(min_samples_split=2,max_depth=6,bootstrap=True,min_samples_leaf=1,n_estimators=100,max_features='auto')
clf.fit(X_train,y_train)
pred=clf.predict(X_test).reshape(-1,1)
cv_result = cross_val_score(clf,pred,y_test,cv=6) 
print('CV Scores: ',cv_result)
print('CV scores average: ',np.sum(cv_result)/6)
# test_data["Survived"]=pred
# test_data[["PassengerId","Survived"]].to_csv("data/predictions/feature_engg_3.csv",index=False)


# In[50]:


# turn run_gs to True if you want to run the gridsearch again.
run_gs = False

if run_gs:
    parameter_grid = {
                 'max_depth' : [4, 6, 8],
                 'n_estimators': [50,10],
                 'max_features': ['sqrt', 'auto', 'log2'],
                 'min_samples_split': [2, 3, 10],
                 'min_samples_leaf': [1, 3, 10],
                 'bootstrap': [True, False],
                 }
    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(n_splits=5)

    grid_search = GridSearchCV(forest,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=cross_validation,
                               verbose=1
                              )

    grid_search.fit(train_data, y)
    model = grid_search
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
    
else: 
    parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, 
                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}
    
    model = RandomForestClassifier(**parameters)
    model.fit(train_data, y)

