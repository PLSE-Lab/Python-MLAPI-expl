#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


data = pd.read_csv("../input/adult.csv")


# ### Exploratory analysis of data

# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data.describe()


# In[ ]:


data.info()


# In[ ]:


columns = data.columns
columns


# In[ ]:


# lets first filter the bad data elements to plot the data and get some insights
labels = data.income
newData = data.drop(labels = ['capital.gain', 'capital.loss', 'income'], axis = 1, inplace = False)


# In[ ]:


newData


# In[ ]:


x= sns.PairGrid(newData, hue='relationship')
# x=x.map(plt.scatter)
x= x.map_diag(plt.hist)
x= x.map_offdiag(plt.scatter)
# plt.figure(figsize=(20,50))
x= x.add_legend()


# In[ ]:


sns.barplot('sex', 'education.num', data = newData)


# In[ ]:


thisArray = np.unique(newData['hours.per.week'].values)
sns.barplot('sex', 'hours.per.week', data = newData[:20], hue = 'hours.per.week')

# np.unique(newData['hours.per.week'].values)


# ## The distribution for income

# In[ ]:


sns.countplot(labels)


# ## Data preprocessing Steps
# 

# Step 1
# - Considering the columns

# In[ ]:


columns = data.columns
columns


# In[ ]:


newData.describe()


# ## Fixing the common nan values
# 
# - Nan values were as ? in data. Hence we fix this with most frequent element in the entire dataset. It generalizes well, as we will see with the accuracy of our classifiers

# In[ ]:


attrib, counts = np.unique(newData['workclass'], return_counts = True)
most_freq_attrib = attrib[np.argmax(counts, axis = 0)]
newData['workclass'][newData['workclass'] == '?'] = most_freq_attrib 

attrib, counts = np.unique(newData['occupation'], return_counts = True)
most_freq_attrib = attrib[np.argmax(counts, axis = 0)]
newData['occupation'][newData['occupation'] == '?'] = most_freq_attrib 

attrib, counts = np.unique(newData['native.country'], return_counts = True)
most_freq_attrib = attrib[np.argmax(counts, axis = 0)]
newData['native.country'][newData['native.country'] == '?'] = most_freq_attrib 


# In[ ]:


newData.head(20)
# Just a check on whether above code works fine.


# In[ ]:


# A lot of features needed to be encoded. Labels such as workplace, relationship, sex, marital.status,
# native.country has no order. Order can be naturally defined, but that would be personally biases.
# In fact, that won't scale for new countries or fresh data.


# In[ ]:


le1 = LabelEncoder()

X = newData.apply(LabelEncoder().fit_transform) 
Y = le1.fit_transform(np.array(labels))


# In[ ]:


Y.shape, X.shape


# In[ ]:


X.head(3)


# In[ ]:



X_encoded = pd.get_dummies(X,columns =["workclass","education","marital.status", "occupation", "relationship",
                                         "race", "sex","native.country"], drop_first=True )


# In[ ]:


Y.shape, X_encoded.shape


# In[ ]:


from sklearn.model_selection import train_test_split as tts
xtrain,xdev,ytrain,ydev = tts(X,Y, random_state = 2, shuffle = True, train_size = 0.7)
xtest,x_dev, ytest,y_dev = tts(xdev,ydev, random_state =2 , shuffle = True, train_size = 0.5)


# In[ ]:


xtrain.shape, xtest.shape, x_dev.shape, ytrain.shape, ytest.shape, y_dev.shape, 


# In[ ]:


# We have the data and the dev set as well as the test set, let us try different algorithms


# In[ ]:


from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.ensemble import BaggingClassifier as BC


# In[ ]:


clf = GBC(n_estimators = 100, max_depth=5) # no overfit, ~85% accuracy
clf.fit(xtrain,ytrain)
score_dev = clf.score(x_dev, y_dev)
score_test = clf.score(xtest, ytest)
score_train = clf.score(xtrain,ytrain)
print(score_train,score_dev,score_test)


# In[ ]:


# clf = SVC()
# clf.fit(xtrain,ytrain)
# score_dev = clf.score(x_dev, y_dev)
# score_test = clf.score(xtest, ytest)
# score_train = clf.score(xtrain,ytrain)
# print(score_train,score_dev,score_test)

# 0.9984204984204984 0.7568065506653019 0.7592137592137592
# Takes a lot of time to run


# In[ ]:


clf = ETC(n_estimators = 50, criterion="entropy", max_depth = 12)
clf.fit(xtrain,ytrain)
score_dev = clf.score(x_dev, y_dev)
score_test = clf.score(xtest, ytest)
score_train = clf.score(xtrain,ytrain)
print(score_train,score_dev,score_test)


# In[ ]:


clf = ABC(n_estimators = 200, learning_rate = 1)
clf.fit(xtrain,ytrain)
score_dev = clf.score(x_dev, y_dev)
score_test = clf.score(xtest, ytest)
score_train = clf.score(xtrain,ytrain)
print(score_train,score_dev,score_test)


# In[ ]:


clf = DTC(max_depth = 7)
clf.fit(xtrain,ytrain)
score_dev = clf.score(x_dev, y_dev)
score_test = clf.score(xtest, ytest)
score_train = clf.score(xtrain,ytrain)
print(score_train,score_dev,score_test)


# In[ ]:


clf = BC(n_estimators = 50, n_jobs = 2)
clf.fit(xtrain,ytrain)
score_dev = clf.score(x_dev, y_dev)
score_test = clf.score(xtest, ytest)
score_train = clf.score(xtrain,ytrain)
print(score_train,score_dev,score_test)


# - Various classifiers were explored. This is clear that classifiers have less to do with the accuracy and more to do with the data presented. Proper presentation of data  is absolute must
# ### End
