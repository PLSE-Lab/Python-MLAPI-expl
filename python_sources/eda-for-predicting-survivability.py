#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import os
data=pd.read_csv('../input/train.csv')


# In[ ]:


data.head()


# In[ ]:


data.describe()


# **38% survived**

# In[ ]:


data.groupby(['Sex'])['Survived'].agg(np.mean)


# In[ ]:


sns.countplot(x='Pclass',data=data,hue='Survived')


# In[ ]:


data.groupby(['Pclass'])['Survived'].agg(np.mean)


# In[ ]:


data.info()


# In[ ]:


'percentage of people survived is more in firstclass'


# In[ ]:


def id_mr(x):
    if 'Miss' in x:
        return True
    else:
        return False


# In[ ]:


data[data['Name'].apply(id_mr)].hist('Age')


# In[ ]:


data['Cabin']=data['Cabin'].replace(np.NAN,'Z')


# In[ ]:


data['Cabin_new']=data['Cabin'].apply(lambda x:x[:1])


# In[ ]:


sns.countplot('Cabin_new',data=data)


# In[ ]:


data.groupby('Cabin_new')['Survived'].agg(np.mean)


# In[ ]:


data.hist('Age',by='Survived',density=True)


# In[ ]:


sns.violinplot(x='Survived',y='Age',data=data)


# Convert Age to categorical data

# In[ ]:


data.hist('Age',by='Survived',density=True)


# In[ ]:


data_new=data.dropna()


# In[ ]:


data_new['Cat_Age']=data_new['Age'].apply(lambda x:int(x/5))


# In[ ]:


sns.countplot('Cat_Age',data=data_new,hue='Survived')


# In[ ]:


data_new.groupby('Cat_Age')['Survived'].agg(np.mean)


# In[ ]:


sns.heatmap(data.corr())


# In[ ]:


sns.distplot(data['Fare'])


# In[ ]:


def norm(mi,ma,x):
    return ((x-mi)*100)/(ma-mi)


# In[ ]:


data.head(1)


# In[ ]:


sns.countplot('SibSp',data=data,hue='Survived')


# In[ ]:


data.groupby(['SibSp'])['Survived'].agg(np.mean)


# In[ ]:


sns.countplot('Parch',data=data,hue='Survived')


# In[ ]:


data.groupby(['Parch'])['Survived'].agg(np.mean)


# In[ ]:





# In[ ]:


data.info()


# In[ ]:


data.groupby(['Embarked'])['Survived'].agg(np.mean)


# In[ ]:


from sklearn.model_selection import train_test_split as split
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


def preprocess(data):
    #features=set(data.columns)
    #features=features-{'PassengerId','Survived','Name','Age','Ticket','Cabin'}
    features=['Pclass', 'Sex', 'Fare', 'Parch', 'Embarked', 'SibSp']
    #features=list(features)
    dat=data[features]
    #print(dat.shape)
    dat['Embarked']=dat['Embarked'].fillna('C')
    dat['Fare']=dat['Fare'].fillna('80')
    gen_onehot_features_sex = pd.get_dummies(dat['Sex'])
    dat=pd.concat([dat, gen_onehot_features_sex],axis=1)
    gen_onehot_features_emb = pd.get_dummies(dat['Embarked'])
    dat=pd.concat([dat, gen_onehot_features_emb],axis=1)
    gen_onehot_features_p = pd.get_dummies(dat['Pclass'])
    dat=pd.concat([dat, gen_onehot_features_p],axis=1)
    dat=dat.drop(['Sex','Embarked','Pclass'],axis=1)

    #dat['Sex']=list(dat['Sex'].factorize()[0])
    #dat['Embarked']=list(dat['Embarked'].factorize()[0])
    #print(dat.shape)
    X=dat.values
    try:
        y=data['Survived'].values
        return X,y
    except:
        return X


# In[ ]:


X,y=preprocess(data)


# In[ ]:


X_train, X_test, y_train, y_test = split(X, y, test_size=0.25, random_state=37)


# In[ ]:


tree=DecisionTreeClassifier(max_depth=3,max_features=9)
tree.fit(X_train,y_train)


# In[ ]:


from sklearn.metrics import accuracy_score

tree_pred = tree.predict(X_test)
accuracy_score(y_test, tree_pred) # 0.94


# In[ ]:


from sklearn.model_selection import GridSearchCV, cross_val_score

tree_params = {'max_depth': range(1,20),
               'max_features': range(1,12)}

tree_grid = GridSearchCV(tree, tree_params,cv=5, n_jobs=-1, verbose=True)

tree_grid.fit(X_train, y_train)


# In[ ]:


tree_grid.best_params_,tree_grid.best_score_


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=100, n_jobs=-1, 
                                random_state=17)
print(np.mean(cross_val_score(forest, X_train, y_train, cv=5)))


# In[ ]:


forest_params = {'max_depth': range(2, 40),
                 'max_features': range(1, 12)}

forest_grid = GridSearchCV(forest, forest_params,
                           cv=5, n_jobs=-1, verbose=True)

forest_grid.fit(X_train, y_train)

forest_grid.best_params_, forest_grid.best_score_


# In[ ]:


from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.25,random_state=17)


knn = KNeighborsClassifier(n_neighbors=10)


# for kNN, we need to scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_holdout_scaled = scaler.transform(X_holdout)
knn.fit(X_train_scaled, y_train)


# In[ ]:


knn_pred = knn.predict(X_holdout_scaled)
accuracy_score(y_holdout, knn_pred)


# In[ ]:


from sklearn.pipeline import Pipeline

knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_jobs=-1))])

knn_params = {'knn__n_neighbors': range(1, 50)}

knn_grid = GridSearchCV(knn_pipe, knn_params,
                        cv=5, n_jobs=-1, verbose=True)

knn_grid.fit(X_train, y_train)

knn_grid.best_params_, knn_grid.best_score_


# In[ ]:


test_data=pd.read_csv('../input/test.csv')


# In[ ]:


test_data.info()


# In[ ]:


X_test_final=preprocess(test_data)


# In[ ]:





# In[ ]:


y_pred_final=tree.predict(X_test_final)
out=pd.DataFrame(index=test_data['PassengerId'])
#out["PassengerId"]=test_data['PassengerId']
out['Survived']=y_pred_final
out.to_csv('gender_submission.csv')


# In[ ]:


out


# In[ ]:


out


# In[ ]:




