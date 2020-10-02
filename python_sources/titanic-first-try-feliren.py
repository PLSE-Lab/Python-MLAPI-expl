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


train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")


# In[ ]:


train.head()


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.describe()


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


train.Survived.value_counts().plot(kind='bar')
plt.title("Survived")


# In[ ]:


train.Survived.value_counts(normalize=True).plot(kind='bar')
plt.title("Survived")


# In[ ]:


plt.scatter(train.Survived,train.Age)
plt.title("Age wrt Survived")


# In[ ]:


train.Pclass.value_counts(normalize=True).plot(kind='bar')
plt.title("Class")


# In[ ]:


#plt.subplot2grid((2,3),(1,0),colspan=2)
for i in [1,2,3]:
    train.Age[train.Pclass == i].plot(kind='kde')
plt.title("Class wrt Age")
plt.legend(('1st','2nd','3rd'))


# In[ ]:


train.Embarked.value_counts(normalize=True).plot(kind='bar')
plt.title('Embarked')


# In[ ]:


train.Survived[train.Sex=='male'].value_counts(normalize=True).plot(kind='bar')
plt.title("Men Survived")


# In[ ]:


train.Survived[train.Sex=='female'].value_counts(normalize=True).plot(kind='bar',color='#FA0000')
plt.title("Women Survived")


# In[ ]:


train.Sex[train.Survived==1].value_counts(normalize=True).plot(kind='bar')
plt.title("Sex of Survived")


# In[ ]:


for i in [1,2,3]:
    train.Survived[train.Pclass == i].plot(kind='kde')
plt.title("Class wrt Survived")
plt.legend(('1st','2nd','3rd'))


# In[ ]:


train.Survived[(train.Sex=='male') & (train.Pclass==1)].value_counts(normalize=True).plot(kind='bar')
plt.title("Rich Men Survived") 


# In[ ]:


train.Survived[(train.Sex=='male') & (train.Pclass==3)].value_counts(normalize=True).plot(kind='bar')
plt.title("Poor Men Survived") 


# In[ ]:


train.Survived[(train.Sex=='female') & (train.Pclass==3)].value_counts(normalize=True).plot(kind='bar')
plt.title("Poor Women Survived") 


# In[ ]:


train.Survived[(train.Sex=='female') & (train.Pclass==1)].value_counts(normalize=True).plot(kind='bar')
plt.title("Rich Women Survived") 


# In[ ]:


train["Hyp"] = 0
train.loc[train.Sex == 'female','Hyp']=1

train["Result"] = 0
train.loc[train.Survived == train['Hyp'],'Result']=1

print(train['Result'].value_counts())


# In[ ]:


def clean_data(data):
    data['Fare'] = data['Fare'].fillna(data['Fare'].dropna().median())
    data['Age'] = data['Age'].fillna(data['Age'].dropna().median())
    
    data.loc[data['Sex'] == 'male','Sex'] = 0
    data.loc[data['Sex'] == 'female','Sex'] = 1
    
    data['Embarked'] = data['Embarked'].fillna('S')
    data.loc[data['Embarked'] == 'S', 'Embarked'] = 0
    data.loc[data['Embarked'] == 'C', 'Embarked'] = 1
    data.loc[data['Embarked'] == 'Q', 'Embarked'] = 2


# In[ ]:


from sklearn import linear_model

clean_data(train)

target = train['Survived'].values
features_name = ['Pclass','Age','Fare','Embarked','Sex','SibSp','Parch']
features = train[features_name].values

glm = linear_model.LogisticRegression()
glm_fit = glm.fit(features,target)

glm_fit.score(features,target)


# In[ ]:


from sklearn import preprocessing

poly = preprocessing.PolynomialFeatures(degree=2)
poly_features = poly.fit_transform(features)

fit_classifier = classifier.fit(poly_features,target)
fit_classifier.score(poly_features,target)


# In[ ]:


from sklearn import tree

decision_tree = tree.DecisionTreeClassifier(random_state=1)
fit_classifier = decision_tree.fit(features,target)
fit_classifier.score(features,target)


# In[ ]:


from sklearn import model_selection
scores = model_selection.cross_val_score(decision_tree,features,target,scoring='accuracy',cv=50)
scores


# In[ ]:


scores.mean()


# In[ ]:


generalized_decision_tree = tree.DecisionTreeClassifier(random_state=1,
                                           max_depth = 7,
                                           min_samples_split = 2)
generalized_decision_tree_fit = generalized_decision_tree.fit(features,target)
print(generalized_decision_tree_fit.score(features,target))

scores = model_selection.cross_val_score(generalized_decision_tree,features,target,scoring='accuracy',cv=50)
print(scores)
print(scores.mean())


# In[ ]:


import graphviz
dot_data = tree.export_graphviz(generalized_decision_tree_fit,
                                feature_names=features_name)
graph = graphviz.Source(dot_data)  
graph 


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(6)
knn_fit = knn.fit(features,target)
knn_fit.score(features,target)


# In[ ]:


clean_data(test)
predictions = glm.predict(test[features_name])
predictions


# In[ ]:


submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})


# In[ ]:


filename = 'Titanic Predictions 1.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)


# In[ ]:




