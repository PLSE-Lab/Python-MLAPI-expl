#!/usr/bin/env python
# coding: utf-8

# To note: I've learned lots of tricks and patterns from Dr.Yassine Ghouzam. He is a great machine learning scientist. Every work out of his magical hands is classic. Thank you very much!

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


import pandas as pd
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
test = pd.read_csv("../input/titanic/test.csv")
train = pd.read_csv("../input/titanic/train.csv")


# In[ ]:


train_len = len(train)
train_len


# In[ ]:


train_wo_Sv = train.drop(labels='Survived', axis= 1)


# Combing train and test datasets together for consistent data cleaning:

# In[ ]:


train_test_total = pd.concat([train_wo_Sv,test], axis = 0)


# In[ ]:


train_test_total.head()


# For Age collumn, first to fill null values:

# In[ ]:


train_test_total['Age'].fillna(value = train_test_total.Age.mean(), inplace = True)


# For Name collumn, to extract the tile using regex, then to replace the low frequent title with 'Rare':

# In[ ]:


import re
train_test_total['Title'] = train_test_total.Name.str.extract('([a-zA-Z]+)\.', expand = True)
train_test_total['Title'] = train_test_total.Title.replace('Mme','Mrs')
train_test_total['Title'] = train_test_total.Title.replace('Mlle','Miss')
train_test_total['Title'] = train_test_total.Title.replace('Ms','Miss')


# In[ ]:


title_counts = train_test_total.Title.value_counts()
least_frequent_title = title_counts[title_counts<=10].index
least_frequent_title

train_test_total['Title'] = train_test_total['Title'].replace(least_frequent_title,'Rare')


# For Fare collumn, to fill na with median value:

# In[ ]:


train_test_total['Fare'] = train_test_total['Fare'].fillna(train_test_total['Fare'].median())


# For Embarked collumn, to fill na with first mode value:

# In[ ]:


train_test_total['Embarked'] = train_test_total['Embarked'].fillna(train_test_total['Embarked'].mode()[0])


# In[ ]:


train_test_total.head()


# In[ ]:


train_test_total.drop(labels=['PassengerId','Cabin','Name','Ticket'], axis = 1, inplace = True)


# In[ ]:


train_test_total.head()


# In[ ]:


train_test_total['Family'] = train_test_total['Parch'] + train_test_total['SibSp'] + 1


# In[ ]:


train_test_total.head()


# To catogerize Age:

# In[ ]:


train_test_total['Age'] = pd.qcut(train_test_total['Age'], 5, labels=['child','young','mid_age','old','very_old'])


# To catogerize Fare:

# In[ ]:


train_test_total['Fare'] = pd.qcut(train_test_total['Fare'], 5, labels=['low','low_medium','medium','high','very_high'])


# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

labelencoder = LabelEncoder()
train_test_total['Sex'] = labelencoder.fit_transform(train_test_total['Sex'])
train_test_total['Embarked'] = labelencoder.fit_transform(train_test_total['Embarked'])
train_test_total['Title'] = labelencoder.fit_transform(train_test_total['Title'])
train_test_total['Age'] = labelencoder.fit_transform(train_test_total['Age'])
train_test_total['Fare'] = labelencoder.fit_transform(train_test_total['Fare'])


# In[ ]:


train_test_total.head()


# In[ ]:


train_test_total = pd.get_dummies(train_test_total, columns= ['Pclass','Embarked','Title','Family','Age','Fare'], drop_first= True, prefix= ['P_','Em_','T_','F_','Age_','Fare_'])


# In[ ]:


train_test_total.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler


# Modelling:****

# To extract the training and test datasets:

# In[ ]:


train_cleaned = train_test_total.iloc[:train_len,:]
test_cleaned = train_test_total.iloc[train_len:,:]
X_train = train_cleaned
Y_train = train.Survived


# To import the various classifier classes:

# In[ ]:


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, GridSearchCV


# To combine the classifiers into the list:

# In[ ]:


random_state = 2
classifiers = []

classifiers.append(RandomForestClassifier(criterion = 'entropy', random_state = random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(criterion = 'entropy', random_state = random_state),learning_rate = 0.1,random_state = random_state))
classifiers.append(GradientBoostingClassifier(random_state = random_state))
classifiers.append(ExtraTreesClassifier(criterion = 'entropy', random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p =2))
classifiers.append(SVC(random_state = random_state))
classifiers.append(SVC(kernel = 'linear',random_state = random_state))
classifiers.append(GaussianNB())


# 1. 1. To fit and transform the training sets using Cross_Val_Score method:

# In[ ]:



cv_results = []
for classifier in classifiers:
    cv_results.append(cross_val_score(estimator = classifier, X = X_train, y = Y_train, scoring = 'accuracy', cv = 10, n_jobs = -1))


# 1. 2. To incoprate the cross_val_score results into cv_mean and cv_std, respectively:

# In[ ]:


cv_mean = []
cv_std = []

for cv_result in cv_results:
    cv_mean.append(cv_result.mean())
    cv_std.append(cv_result.std())
    
cv_vis = pd.DataFrame({'CrossValMean':cv_mean,'CrossValError': cv_std ,'Algorithm':['RandomForest','Adaboost','Gradientboost','Extratrees',
                                                            'LinearDiscriminat','LogisticReg','KNeighbor','Kernel_SVC','SVC','GaussianNB']})


# 1. 3. To visualize the accuracies and corresponding algorithm:

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize= (8,5))
sns.barplot(y = 'Algorithm',x = 'CrossValMean', data= cv_vis, orient='h', **{'xerr': cv_std})
plt.xlabel('Mean Accuracy', fontsize = 15)
plt.ylabel('Algorithm', fontsize = 15)
plt.yticks(rotation = 35)
plt.show()


# 2. 1. To apply GridSearch on best performing models: SVC, LogisticRegression, LinearDiscrimination, Gradientboost:

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


# Gradient boosting tunning

GBC_classifier = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01], 
              }

gsGBC = GridSearchCV(GBC_classifier,param_grid = gb_param_grid, cv=10, scoring="accuracy", n_jobs= 4, verbose = 1)

gsGBC.fit(X_train,Y_train)

GBC_best = gsGBC.best_estimator_

# Best score
gsGBC.best_score_


# In[ ]:


# Linear Discriminant Analysis tunning

LDA_classifier = LinearDiscriminantAnalysis()
gb_param_grid = {
              }

gsLDA = GridSearchCV(LDA_classifier,param_grid = gb_param_grid, cv=10, scoring="accuracy", n_jobs= 4, verbose = 1)

gsLDA.fit(X_train,Y_train)

LDA_best = gsLDA.best_estimator_

# Best score
gsLDA.best_score_


# In[ ]:


# Logistic Regression tunning

LR_classifier = LogisticRegression()
gb_param_grid = {
                  'C': [0.1,1, 10, 50, 100,]
              }

gsLR = GridSearchCV(LR_classifier,param_grid = gb_param_grid, cv=10, scoring="accuracy", n_jobs= 4, verbose = 1)

gsLR.fit(X_train,Y_train)

LR_best = gsLR.best_estimator_

# Best score
gsLR.best_score_


# In[ ]:


# Kernel SVC tunning

KSVC_classifier = SVC(probability=True)
gb_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [0.1,1, 10, 50, 100,]
              }

gsKSVC = GridSearchCV(KSVC_classifier,param_grid = gb_param_grid, cv=10, scoring="accuracy", n_jobs= 4, verbose = 1)

gsKSVC.fit(X_train,Y_train)

KSVC_best = gsKSVC.best_estimator_

# Best score
gsKSVC.best_score_


# Ensemble Voting:

# In[ ]:


votingC = VotingClassifier(estimators=[('gbc', GBC_best), ('lda', LDA_best),
('lr', LR_best), ('ksvc',KSVC_best)], voting='soft', n_jobs=4)

votingC = votingC.fit(X_train, Y_train)


# To predict and submit results:

# In[ ]:


test_Survived = pd.Series(votingC.predict(test_cleaned), name="Survived")

results = pd.concat([test.PassengerId,test_Survived],axis=1)

results.to_csv("ensemble_python_voting.csv",index=False)

