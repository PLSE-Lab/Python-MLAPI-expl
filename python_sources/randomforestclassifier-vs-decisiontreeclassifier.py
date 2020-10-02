#!/usr/bin/env python
# coding: utf-8

# The Titanic
# Using information about passengers of the Titanic, we are interested in building a model based on a DecisionTreeClassifier and RandomForestClassifier to say something about the chances of surviving the disaster.
# 
# The training data provided contains 891 records with the following attributes:
# 
# Survived: 0= No; 1 = Yes
# 
# Pclass: Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
# 
# Name: Passenger name
# 
# Sex: (female; male)
# 
# Age: Passenger age
# 
# SibSp: Number of Siblings/Spouses Aboard
# 
# Parch: Number of Parents/Children Aboard
# 
# Ticket:Ticket Number
# 
# Fare: Passenger Fare
# 
# Cabin: Cabin
# 
# Embarked: Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

# In[63]:


import numpy as np # linear algebra
np.random.seed(42)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

dataset = pd.read_csv("../input/train.csv")

dataset.head()


# In[64]:


dataset.info()


# In[65]:


dataset.describe()


# In[66]:


dataset.corr()


# lets visualize correlation

# In[67]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[68]:


sns.heatmap(dataset.corr(),annot=True)
plt.show()


# In[69]:


Sex_pct = pd.crosstab(
    dataset['Sex'].astype('category'),
    dataset['Survived'].astype('category'),
    margins=True,
#     normalize=True
)
Sex_pct


# In[70]:


sns.barplot('Sex','Survived',data=dataset)
plt.show()


# In[71]:


dataset.Sex.value_counts()


# In[72]:


sns.countplot(dataset.Age.value_counts())
plt.show()


# In[73]:


sns.pairplot(dataset)
plt.show()


# In[74]:


sns.countplot(dataset.Sex.value_counts())
plt.show()


# Find missing value and remove it

# In[75]:


dataset.isnull().sum(axis=0)


# In[76]:


dataset.dropna(inplace=True)


# Make data normal or in numeric so model can learn better

# In[77]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset.Embarked = le.fit_transform(dataset.Embarked)
dataset.Sex = le.fit_transform(dataset.Sex)
dataset.head()


# In[78]:


X,y = dataset[['Pclass','Sex','Age','Embarked']],dataset['Survived']


# In[79]:


X.head()


# In[80]:


y.head()


# In[81]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)


# In[82]:


from sklearn.metrics import accuracy_score
y_pred = tree_clf.predict(X_test)
print(tree_clf.__class__.__name__, accuracy_score(y_test, y_pred))
print(f'Classification Report for {tree_clf.__class__.__name__}')
print(classification_report(y_test, y_pred))
print('*'*60)


# In[83]:


bag_clf = BaggingClassifier(DecisionTreeClassifier(random_state=42), 
                            n_estimators=500,
                            bootstrap=True, n_jobs=-1)
bag_clf.fit(X_train, y_train)


# ![](http://)Bagging claasifier with DecisionTreeClassifier will improve the accuracy

# In[84]:


y_pred = bag_clf.predict(X_test)
print(bag_clf.__class__.__name__, accuracy_score(y_test, y_pred))
print(f'Classification Report for {bag_clf.__class__.__name__}')
print(classification_report(y_test, y_pred))
print('*'*60)


# ****Random Forests****

# In[85]:


from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(X_train, y_train)


# In[86]:


y_pred_rf = rnd_clf.predict(X_test)

accuracy_score(y_test, y_pred_rf)


# You are able to notice that random forest and decision tree with bagging classifier will give somewhere same accuracy.

# In[87]:


bag_clf = BaggingClassifier(
    DecisionTreeClassifier(splitter='random', random_state=42),
    n_estimators=500, n_jobs=-1
)
bag_clf.fit(X_train, y_train)


# In[88]:


y_pred = bag_clf.predict(X_test)

accuracy_score(y_test, y_pred)


# random forest and decision tree with bagging classifier predicition simmilarity

# In[89]:


np.sum(y_pred == y_pred_rf) / len(y_pred)  # almost identical predictions


# In[90]:


output = pd.DataFrame(X_test)
output['y_pred'] = y_pred


# In[91]:


output.head()

