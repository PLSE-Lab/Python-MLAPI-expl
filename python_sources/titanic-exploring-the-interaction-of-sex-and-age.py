#!/usr/bin/env python
# coding: utf-8

# ## Titanic - Exploring interaction of Sex and Age

# Lets take a deeper look at Age and its connection with survival. Most of the solutions available here used bins of age without much evaluation. We will look at this relationship and check if we can handle Age better.
# 
# I am thankful to **headsortail**  for the kernal published here (https://www.kaggle.com/headsortails/pytanic).

# In[ ]:


import pandas as pd
import numpy as np
from scipy import stats
import sklearn as sk
import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic
from fancyimpute import  KNN

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import svm
import xgboost as xgb
import lightgbm as lgb
from mlxtend.classifier import StackingClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

sns.set(style='white', context='notebook', palette='deep')

import os     
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

get_ipython().run_line_magic('matplotlib', 'inline')


# Let's read both Train and Test and combine them for cleaning.'

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
tot= pd.concat([train.drop('Survived',1),test])
survived = train['Survived']


# In[ ]:


y=train['Survived']
X=train.iloc[:, 1:10]
X.head()


# In[ ]:


pd.crosstab(train.Sex, train.Survived) 


# In[ ]:


tot.head(3)


# Let's convert Sex into a dummy variable for further analysis.

# In[ ]:


tot['Sex'] = tot['Sex'].map({'male': 0, 'female': 1})


# In[ ]:


tot.drop(['Ticket','Cabin', 'Embarked'], axis=1,inplace=True)
tot.describe()


# There are missing values in Age and lets impute it. Since our focus is on Age, we will use knn based imputation as it will use all the available inforation in the data. We will use the entire dataset fot this (test+train). 
# 
# There is one missing value for Fare. This too will be taken care.
# 
# We have used 3 neighbour solution. In fact this can be tuned to get the best.

# In[ ]:


tot_num = tot.select_dtypes(include=[np.number])
tot_numeric = tot.select_dtypes(include=[np.number]).as_matrix()
tot_filled = pd.DataFrame(KNN(3).complete(tot_numeric))
tot_filled.columns = tot_num.columns
tot_filled.index = tot_num.index


# In[ ]:


print(tot_filled.info(), tot_filled.describe())
tot=tot_filled


# Now the data is fine. We will now focus on Sex and Age.
# 
# **Let's take a look at the ditribution of Age**

# In[ ]:


test = tot.iloc[len(train):]
train = tot.iloc[:len(train)]
train['Survived'] = survived
surv = train[train['Survived']==1]
nosurv = train[train['Survived']==0]


# In[ ]:


y_train=train['Survived']
X_train=train.iloc[:, 1:7]
X_tree = X_train[['Sex', 'Age']]


# In[ ]:


plt.figure(figsize=[15,10])
sns.distplot(train['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='blue', axlabel='Age')


# Most of the passengers are between age 20 and 40 years age. How about their survival??

# In[ ]:


msurv = train[(train['Survived']==1) & (train['Sex']==0)]
fsurv = train[(train['Survived']==1) & (train['Sex']==1)]
mnosurv = train[(train['Survived']==0) & (train['Sex']==0)]
fnosurv = train[(train['Survived']==0) & (train['Sex']==1)]


# In[ ]:


plt.figure(figsize=[15,10])
sns.distplot(surv['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='blue', axlabel='Age')
sns.distplot(nosurv['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='red')


# **Obviously the relationship is complex.** At young age (less than about 18), it seems positive as survival is higher. However, in between 18 and 30 (approx), survival is lower (a contradiction). Above 30, it seems there is no significant difference. We are not sure about the exact age  at which these flipping is happening. Lets take a look at the interaction of Sex on this relationship.

# In[ ]:


plt.figure(figsize=[15,10])
plt.subplot(211)
sns.distplot(fsurv['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='blue')
sns.distplot(fnosurv['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='red',
            axlabel='Female Age')
plt.subplot(212)
sns.distplot(msurv['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='blue')
sns.distplot(mnosurv['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='red',
            axlabel='Male Age')


# Looks like there is interesting difference between Male and Female. The overall concluion we made in the previous section is mostly driven by Male passengers; mainly becuase they were in the majority.
# 
# Now the best way to formalize this intereaction is to conduct a Decision Tree analysis. This analysis will apply approproate parameters and bring out optimum cutoffs.

# In[ ]:


from sklearn import tree
model = tree.DecisionTreeClassifier(min_samples_split=10, max_depth=3, min_samples_leaf=50)
model


# We will have to iterate few times varying the parameters to get the Tree which make sense. Mostly we will have to adjust the min_samples_leaf and min_samples_split to get a tree that is well balanced in terms of height and width.

# In[ ]:


model.fit(X_tree, y_train)


# In[ ]:


y_predict = model.predict(X_tree)
from sklearn.metrics import accuracy_score
accuracy_score(y_train, y_predict)


# Overall accuracy is 0.78;  which is OK considering that we used only two features.

# In[ ]:


import graphviz
dot_data = tree.export_graphviz(model, feature_names=X_tree.columns, out_file=None, filled=True, rounded=True, special_characters=True)
graphviz.Source(dot_data)


# Quite interesting. As expected, Sex plays an important role as influnce of age is different for Male and Female. The Tree algorithm is doing a wonderful job of cutting age where the behaviour changes. Now let's mark these on the histplot. 
# 
# Black vertical lines mark the cutoff's identified by the Tree algorithm.

# In[ ]:


plt.figure(figsize=[15,10])
plt.subplot(211)
plt.axvline(x=24, color='black')
plt.axvline(x=31, color='black')
plt.axvline(x=41, color='black')
sns.distplot(fsurv['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='blue')
sns.distplot(fnosurv['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='red',axlabel='Female Age')
plt.subplot(212)
plt.axvline(x=16, color='black')
plt.axvline(x=25, color='black')
sns.distplot(msurv['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='blue')
sns.distplot(mnosurv['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='red', axlabel='Male Age')


# The vertical lines (the cutoffs identified by Tree algorithm) is at places where the survival behavior is changing. At the highest level, it is 16 years for Male and and 31 years for Female. 
# 
# Interestingly Male passengers below 16 only got the advantage of being a child. Above this survival dropped as they are being percieved as 'adults'. It is bit confusing why survival is marginally lower for young women below 24 years. Above 24, it s highher till 40 and then there is no difference.
# 
# This analysis established that survival depends on Age and Sex is an interacting feature. Hence, at modeling stage we should consider this.
# 
# **Next step would be to create dummies representing these and try it in the model**

# 
