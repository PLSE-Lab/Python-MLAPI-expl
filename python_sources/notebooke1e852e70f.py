#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

##seems like labelencoder doesn't allow you to replace labels and preserve order of magnitude
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/survey.csv')


# In[ ]:


#Have a look at the data you have
list(df)


# In[ ]:


df.head()


# In[ ]:


df.family_history = le.fit_transform(df.family_history) 
df.mental_health_consequence = le.fit_transform(df.mental_health_consequence)
df.phys_health_consequence = le.fit_transform(df.phys_health_consequence)
df.coworkers = le.fit_transform(df.coworkers)
df.supervisor = le.fit_transform(df.supervisor)
df.mental_health_interview = le.fit_transform(df.mental_health_interview)
df.phys_health_interview = le.fit_transform(df.phys_health_interview)
df.mental_vs_physical = le.fit_transform(df.mental_vs_physical)
df.obs_consequence = le.fit_transform(df.obs_consequence)
df.remote_work = le.fit_transform(df.remote_work)
df.tech_company = le.fit_transform(df.tech_company)
df.benefits = le.fit_transform(df.benefits)
df.care_options = le.fit_transform(df.care_options)


# In[ ]:


df.care_options.unique()


# In[ ]:


#workaround to use labelencoder on NaNs. Yes having thought about it now, there are better ways :/
df.self_employed[pd.isnull(df.self_employed)]='NaN'
df.loc[df.self_employed[pd.isnull(df.self_employed)],'self_employed']='NaN'
df.self_employed = le.fit_transform(df.self_employed)


# In[ ]:


#NaNs need to be converted to string before they can be used in the encoder.
df['self_employed'][pd.isnull(df['self_employed'])]='NaN'

#df['self_employed'].loc[[pd.isnull(df['self_employed'])],'self_employed']  = 'NaN'
#df.self_employed = le.fit_transform(df.self_employed)
#df.loc[df['no_employees']=='1-5',['no_employees']]=1


# In[ ]:


df.loc[df['self_employed'][pd.isnull(df['self_employed'])],'self_employed']  = 'NaN'


# In[ ]:


#A better way to deal with nulls!
# Now change comments column to flag whether or not respondent made additional comments
df.loc[df['comments'].isnull(),['comments']]=0 # replace all no comments with zero
df.loc[df['comments']!=0,['comments']]=1 # replace all comments with a flag 1


# Preserve Scale in some of the features
# --------------------------------------

# In[ ]:


df['leave'].replace(['Very easy', 'Somewhat easy', "Don\'t know", 'Somewhat difficult', 'Very difficult'], 
                     [1, 2, 3, 4, 5],inplace=True) 
df['work_interfere'].replace(['Never','Rarely','Sometimes','Often'],[1,2,3,4],inplace=True)


#From assessing the unique ways in which gender was described above, the following script replaces gender on
#a -2 to 2 scale:
#-2:male
#-1:identifies male
#0:gender not available
#1:identifies female
#2: female.

#note that order of operations matters here, particularly for the -1 assignments that must be done before the
#male -2 assignment is done

df.loc[df['Gender'].str.contains('F|w', case=False,na=False),'Gender']=2
df.loc[df['Gender'].str.contains('queer/she',case=False,na=False),'Gender']=1
df.loc[df['Gender'].str.contains('male leaning',case=False,na=False),'Gender']=-1
df.loc[df['Gender'].str.contains('something kinda male',case=False,na=False),'Gender']=-1
df.loc[df['Gender'].str.contains('ish',case=False,na=False),'Gender']=-1
df.loc[df['Gender'].str.contains('m',case=False,na=False),'Gender']=-2
df.loc[df['Gender'].str.contains('',na=False),'Gender']=0


# In[ ]:


df.loc[df['no_employees']=='1-5',['no_employees']]=1
df.loc[df['no_employees']=='6-25',['no_employees']]=2
df.loc[df['no_employees']=='26-100',['no_employees']]=3
df.loc[df['no_employees']=='100-500',['no_employees']]=4
df.loc[df['no_employees']=='500-1000',['no_employees']]=5
df.loc[df['no_employees']=='More than 1000',['no_employees']]=6


# Some features can just go
# -------------------------

# In[ ]:


# Feature selection
drop_elements = ['Timestamp','Country','state']
df = df.drop(drop_elements, axis = 1)


# In[ ]:


df.columns


# Set up features and dependent variable
# --------------------------------------

# In[ ]:



X = df
X = df.drop('treatment',axis=1)
y = df['treatment']
y = le.fit_transform(y) # yes:1 no:0


# Apportion data into train and test
# ----------------------------------

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =         train_test_split(X, y, test_size=0.20, random_state=1)


# Set up pipeline of preprocessing steps
# --------------------------------------
# 
# In this case for a logistic regression with scaled data and PCA on the first 2 components
# 

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe_lr = Pipeline([('scl', StandardScaler()),
            ('pca', PCA(n_components=2)),
            ('clf', LogisticRegression(penalty='l2', random_state=1))])
#although l2 regularisation specified, this is the default set anyway. C parameter uses a default of 1


# In[ ]:


pipe_lr.get_params


# Now try cross validation method on your logistic regression model
# -----------------------------------------------------------------
# 
# Note that this is a 10-fold stratified cross validation as indicated by cv = 10
# 

# In[ ]:


X.head()


# In[ ]:





# In[ ]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(estimator=pipe_lr, 
                         X=X_train, 
                         y=y_train, 
                         cv=10,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('mean CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


# Note that the accuracy of this model is very low. Which is not surprising since you only have 3 features.

# Use learning curve to determine how well your model performs with increasing data
# ------------------------------------------------------------------------
# 
# i.e. does it suffer from high bias or high variance
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


# In[ ]:


train_sizes, train_scores, cv_scores =                learning_curve(estimator=pipe_lr, 
                X=X_train, 
                y=y_train, 
                train_sizes=np.linspace(0.1, 1.0, 10), 
                cv=10,
                n_jobs=1)
    
#The combination of train_sizes and cv set up the incremements to your data set. 
#For instance cv=10 divides data into 10 stratified folds 
#1/10 is the cross validation set
#9/10 is the training set
# That 9/10 is further divided into increasing train_sizes as determined by linspace
# see cell below for the numbers

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
cv_mean = np.mean(cv_scores, axis=1)
cv_std = np.std(cv_scores, axis=1)

plt.plot(train_sizes, train_mean, 
         color='blue', marker='o', 
         markersize=5, label='training accuracy')

plt.fill_between(train_sizes, 
                 train_mean + train_std,
                 train_mean - train_std, 
                 alpha=0.15, color='blue')

plt.plot(train_sizes, cv_mean, 
         color='green', linestyle='--', 
         marker='s', markersize=5, 
         label='validation accuracy')

plt.fill_between(train_sizes, 
                 cv_mean + cv_std,
                 cv_mean - cv_std, 
                 alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.4, 1.0])
plt.tight_layout()
# plt.savefig('./figures/learning_curve.png', dpi=300)
plt.show()


# Judging by how low the accuracy is for both training and validation data, conclude that model suffers both from high bias and variance.

# In[ ]:


print(train_sizes)
print(len(X_train))
print(np.linspace(0.1, 1.0, 10))
print(train_scores.shape)
print(cv_scores.shape)


# Pick Regularisation Hyperparameter C for logistic regression model
# ====================================

# In[ ]:


from sklearn.model_selection import validation_curve

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
train_scores, test_scores = validation_curve(
                estimator=pipe_lr, 
                X=X_train, 
                y=y_train, 
                param_name='clf__C', 
                param_range=param_range,
                cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean, 
         color='blue', marker='o', 
         markersize=5, label='training accuracy')

plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, alpha=0.15,
                 color='blue')

plt.plot(param_range, test_mean, 
         color='green', linestyle='--', 
         marker='s', markersize=5, 
         label='validation accuracy')

plt.fill_between(param_range, 
                 test_mean + test_std,
                 test_mean - test_std, 
                 alpha=0.15, color='green')

plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.55, 0.65])
plt.tight_layout()
# plt.savefig('./figures/validation_curve.png', dpi=300)
plt.show()


# Based on above image, a good value for c is 0.1 # when data is based on 3 features 'Age','Gender','no_employees','comments'

# Now run Gridsearch to pick the best parameters for a selection of different models
# ========================================================================
# 
# SVM Model
# ---------

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

pipe_svc = Pipeline([('scl', StandardScaler()),
            ('clf', SVC(random_state=1))])

param_range = [0.01, 0.1, 1.0, 10.0, 100.0]

param_grid = [{'clf__C': param_range, 
               'clf__kernel': ['linear']},
                 {'clf__C': param_range, 
                  'clf__gamma': param_range, 
                  'clf__kernel': ['rbf']}]

#lili this bit is the inner loop
gs = GridSearchCV(estimator=pipe_svc, 
                            param_grid=param_grid, #this bit does the grid search of the parameter space i.e. linear/rbf and parameter tuning
                            scoring='accuracy', 
                            cv=2,
                            n_jobs=-1)

# Note: Optionally, you could use cv=2 
# in the GridSearchCV above to produce
# the 5 x 2 nested CV that is shown in the figure.

gs = gs.fit(X_train, y_train)
print(gs.best_score_) #whilst these numbers are interesting, they are not the outer loop cross-validation as
                      #below so will not be quoted as the training accuracy. 
print(gs.best_params_)#

#lili this bit is the outer loop
scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


print(scores)

#gs = gs.fit(X_train, y_train)
#clf = gs.best_estimator_
#clf.fit(X_train, y_train)
#print('Test accuracy: %.3f' % clf.score(X_test, y_test))


# In[ ]:


gs.cv_results_ #test score is a little bit confusing here. Remember that test here actually refers 
               #to cross-validation as gs forms the inner loop of the cross-validation process
               #set up in the previous cell.


# Logistic regression model
# -------------------------

# In[ ]:


param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'clf__C': param_range}]

#lili this bit is the inner loop
gs = GridSearchCV(estimator=pipe_lr, 
                            param_grid=param_grid, #this bit does the grid search of the parameter space i.e. linear/rbf and parameter tuning
                            scoring='accuracy', 
                            cv=2,
                            n_jobs=-1)

# Note: Optionally, you could use cv=2 
# in the GridSearchCV above to produce
# the 5 x 2 nested CV that is shown in the figure.

#lili this bit is the outer loop
scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

#gs = gs.fit(X_train, y_train)
#print(gs.best_score_)
#print(gs.best_params_)


# Decision Tree Model
# -------------------

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0), 
                            param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}], 
                            scoring='accuracy', 
                            cv=2)
scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)


# In[ ]:


Clearly Decision Trees is the winner here with 3 features # 'Age','no_employees','comments'
svm is marginally better with 4 features # 'Age','Gender','no_employees','comments'


# In[ ]:




