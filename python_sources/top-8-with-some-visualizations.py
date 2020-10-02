#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# read in train and test data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


# info for null values
train.info()


# In[ ]:


# need to dropna on embarked before we append
train.dropna(subset=['Embarked'], inplace=True)


# In[ ]:


test.info()


# In[ ]:


# Append the test data to the train data so both are manipulated the same way
data= train.append(test, ignore_index=True)


# In[ ]:


# drop ID but we will need it later for submission
#data = data.drop('PassengerId', axis=1)
train_id = train['PassengerId']
test_id = pd.DataFrame(test['PassengerId'])


# In[ ]:


data.describe()


# In[ ]:


data.head()


# #### Creating a new column, 'title' to more accurately estimate the ages that are NaN

# In[ ]:


data['title'] = data.Name.map(lambda x: x.split(",")[1].split('.')[0].strip())


# In[ ]:


data.head(6)


# In[ ]:


data.groupby('title')['Age'].mean()


# In[ ]:


data.groupby('title')['Age'].count()


# #### option to fill in age here:

# In[ ]:


data['Age'] = data.groupby('title').transform(lambda x: x.fillna(x.mean()))


# In[ ]:


# Checking the data where age is null
data[data.Age.isnull()]


# In[ ]:


fig = plt.figure(figsize=(16,12))
ax1 = fig.add_subplot(111)
sns.boxplot(x = 'Embarked', y='Fare', hue='Pclass', data=data, ax=ax1)
ax1.set_title('Fare based on Embarked and separated by Pclass')


# In[ ]:


fig = plt.figure(figsize=(22,14))
ax1 = fig.add_subplot(211)
sns.boxplot(x = 'title', y='Age', data=data, ax=ax1)
ax1.set_title('Boxplot of age based on Title')
ax2 = fig.add_subplot(212)
sns.boxplot(x='title', y='Fare', data=data, ax=ax2)
ax2.set_title('Boxplot of Fare paid based on Title');


# In[ ]:


#sns.catplot(x='Age', y='Embarked', hue='Pclass', col='Survived', height=6, kind='point', data=data)


# In[ ]:


#heatmap to show correlation of all the numeric variables
plt.figure(figsize=(16,8))
sns.heatmap(data.corr(), annot=True)


# In[ ]:


#Scatter plot of fare and Age
plt.figure(figsize=(16,8))
plt.scatter(x='Fare', y='Age', data=data)
plt.xlabel('Fare($)', fontsize=13)
plt.ylabel('Age', fontsize=13)
plt.title('Fare and Age Scatter', fontsize=16);


# In[ ]:


'''fig = plt.figure(figsize=(16,18))
ax1= fig.add_subplot(111)
sns.relplot(x='title', y="Fare", hue="Survived", size="Age",
            sizes=(40, 540), alpha=.5,
            height=6, data=data,ax=ax1)'''


# In[ ]:


# 3d Plot that shows Fare along with survived and Age
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111, projection='3d')
xs = data['Fare']
ys = data['Survived']
zs = data['Age']
ax.scatter(xs,ys,zs)
ax.set_xlabel('Fare')
ax.set_ylabel('Survived')
ax.set_zlabel('Age')
plt.show()


# In[ ]:


sns.set(style="whitegrid", palette="muted")
plt.figure(figsize=(16,12))
sns.swarmplot(x="Pclass", y="Fare", hue="Survived",
              palette=["r", "c", "y"], data=data)


# In[ ]:


#This came from the seaborn gallery. It's a great visual
sns.set(style="whitegrid", palette="muted")
plt.figure(figsize=(18,10))
sns.catplot(x='Pclass', y='Survived', hue='Sex', data=data, kind='bar', height=8)


# In[ ]:


data.info()


# ### There is a value missing in fare. So I'm going to take the median fare price in the dataset and fill in that value  

# In[ ]:


data['Fare'] = data['Fare'].fillna(data['Fare'].median())


# In[ ]:


data.info()


# In[ ]:


# dummies=pd.get_dummies(data, columns=['Embarked', 'Pclass', 'Sex'], drop_first=True)
dummies = pd.get_dummies(data[['Sex','Embarked']],drop_first=True)
data = data.join(dummies)


# In[ ]:


pclass = pd.get_dummies(data['Pclass'], drop_first=True)
data= data.join(pclass)


# In[ ]:


data.drop(['Cabin', 'Ticket', 'Name','Sex', 'Embarked','title', 'Pclass'], axis=1, inplace=True)


# In[ ]:


missing = data[data['Survived'].isnull()]
filled = data[data['Survived'].notnull()]


# In[ ]:


filled= filled.drop('PassengerId', axis=1)
missing = missing.drop('PassengerId', axis=1)


# In[ ]:


plt.hist(data['Age'].dropna(), bins=100);


# In[ ]:


data.head()


# ### Modeling

# #### Decision tree, Random Forest, Gradient Boosting Classifier

# In[ ]:


from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(filled.drop('Survived', axis=1), filled['Survived'], test_size=.33, random_state=42)


# In[ ]:


# Specifying the algorithms 
dt = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=605, min_samples_split=4,random_state=22)
# best score is with the parameters commented out right here:
#gbc = GradientBoostingClassifier(n_estimators= 140,max_depth=4,random_state=32)
gbc = GradientBoostingClassifier(n_estimators= 120,max_depth=4,random_state=32)# lower learning rate when increasing n_estimators


# In[ ]:


algorithms = [dt,rf,gbc]
names = ['Decision Tree', 'Random Forest', 'Gradient Boosting']


# In[ ]:


def tDMassess():
    #fit the data
    for i in range(len(algorithms)):
        algorithms[i] = algorithms[i].fit(X_train,y_train)
    accuracy_train =[]
    accuracy_test=[]
    for i in range(len(algorithms)):
        #print(i) 
        accuracy_train.append(accuracy_score(filled['Survived'], algorithms[i].predict(filled.drop('Survived', axis = 1))))
        #print(accuracy)
        accuracy_test.append(accuracy_score(y_test, algorithms[i].predict(X_test)))
        #print(f1)
        #print('next loop')
    metrics = pd.DataFrame(columns =['Accuracy_Train', 'Accuracy_Test'], index=names)#we defined index=names above, where we defined the algorithms
    metrics['Accuracy_Train'] = accuracy_train
    metrics['Accuracy_Test'] = accuracy_test
    return metrics


# In[ ]:


tDMassess()


# ### Now to use the best model to Submit

# In[ ]:


y_pred = gbc.predict(missing.drop('Survived', axis=1))


# In[ ]:


test_id['Survived'] = y_pred.astype('int')


# In[ ]:


y_test = y_test.astype('int')
y_pred = y_pred.astype('int')


# In[ ]:


y_pred = gbc.predict(missing.drop('Survived', axis=1))
test_id.to_csv('submission.csv', index=False)

