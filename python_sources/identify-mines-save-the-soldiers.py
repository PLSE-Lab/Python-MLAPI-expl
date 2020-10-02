#!/usr/bin/env python
# coding: utf-8

# # Connectionist Bench (Sonar, Mines vs. Rocks) Data Set
# 
# ## Source:
# 
# ### The data set was contributed to the benchmark collection by Terry Sejnowski, now at the Salk Institute and the University of California at San Deigo. The data set was developed in collaboration with R. Paul Gorman of Allied-Signal Aerospace Technology Center.
# 
# 
# ## Data Set Information:
# 
# ### The file "sonar.mines" contains 111 patterns obtained by bouncing sonar signals off a metal cylinder at various angles and under various conditions. The file "sonar.rocks" contains 97 patterns obtained from rocks under similar conditions. The transmitted sonar signal is a frequency-modulated chirp, rising in frequency. The data set contains signals obtained from a variety of different aspect angles, spanning 90 degrees for the cylinder and 180 degrees for the rock.
# 
# ### Each pattern is a set of 60 numbers in the range 0.0 to 1.0. Each number represents the energy within a particular frequency band, integrated over a certain period of time. The integration aperture for higher frequencies occur later in time, since these frequencies are transmitted later during the chirp.
# 
# ### The label associated with each record contains the letter "R" if the object is a rock and "M" if it is a mine (metal cylinder). The numbers in the labels are in increasing order of aspect angle, but they do not encode the angle directly.
# 
# ## Source: University of California ML Repository
# ### https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)

# ### This is a binary classification problem
# ### 'R' is for the class Rock and 'M' is for class Mine

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, recall_score


# In[ ]:


df = pd.read_csv('/kaggle/input/mines-vs-rocks/sonar.all-data.csv', header=None)


# In[ ]:


df


# In[ ]:


df.describe().T


# In[ ]:


df.info()


# In[ ]:


fig, axs = plt.subplots(figsize=(10, 8))
sns.countplot(df[60], ax=axs)
plt.show()


# In[ ]:


df.hist(figsize=(15, 10))


# In[ ]:


plt.figure(figsize=(15, 10))
sns.heatmap(df.corr())


# In[ ]:


X = df.drop(columns=60).values
y = df[60]
y = y.map({'R' : 0, 'M' : 1}).values


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


# In[ ]:


classifiers = [('KNN', KNeighborsClassifier()), 
               ('SVC', SVC()), 
               ('GPC', GaussianProcessClassifier()), 
               ('DTC', DecisionTreeClassifier()), 
               ('RFC', RandomForestClassifier()), 
               ('MLPC', MLPClassifier()), 
               ('ABC', AdaBoostClassifier()), 
               ('GNB', GaussianNB()), 
               ('QDA', QuadraticDiscriminantAnalysis()), 
               ('LDA', LinearDiscriminantAnalysis()), 
               ('LR', LogisticRegression())]


# In[ ]:


results = []
names = []
scoring = 'accuracy'

for name, classifier in classifiers:
    kfold = KFold(n_splits=10, shuffle=True)
    cv_score = cross_val_score(estimator=classifier, X=X_train, y=y_train, scoring=scoring)
    results.append(cv_score)
    names.append(name)
    print('Classifier: {}, Mean Accuracy: {}, StDev: {}'.format(name, cv_score.mean(), cv_score.std()))


# In[ ]:


fig, axs = plt.subplots(figsize=(15, 10))
axs.boxplot(results)
axs.set_xticklabels(names, fontdict={'size' : 18})
plt.show()


# In[ ]:


scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[ ]:


results_scaled_data = []
names = []

for name, model in classifiers:
    kfold = KFold(n_splits=10, shuffle=True)
    cv_score = cross_val_score(estimator=model, X=X_train_scaled, y=y_train, scoring='accuracy')
    results_scaled_data.append(cv_score)
    names.append(name)
    print('Model: {}, Mean Score: {}, Score StDev: {}'.format(name, cv_score.mean(), cv_score.std()))


# In[ ]:


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 12), sharex=True, sharey=True)
axs[0].boxplot(results_scaled_data)
axs[0].set_title('Standardised Data', fontdict={'size' : 18})
axs[0].set_xticklabels(names)
axs[0].grid(linestyle='dashed', linewidth=0.2, color='black')

axs[1].boxplot(results)
axs[1].set_title('Non-Standardised Data', fontdict={'size' : 18})
axs[1].grid(linestyle='dashed', linewidth=0.2, color='black')


# ### It looks like standardised data is producing better results, as the accuracy score distributions are a bit higher than the non-standardised data. 
# ## Next, we pick three best performing models, GPC (Gaussian Process Classifier), MPLC (MPL Classifier) and SVC (Support Vector Classifier), and attempt to improve results further by parameter tuning.

# # Tuning SVC

# In[ ]:


svc = SVC()

param_grid = {'C' : [1, 10, 100, 1000], 'kernel' : ['linear', 'rbf', 'sigmoid'], 
              'gamma' : ['scale', 'auto', 1.0, 0.1, 0.01, 0.001, 0.0001]}

kfold = KFold(n_splits=10, shuffle=True)
gsearch = GridSearchCV(estimator=svc, param_grid=param_grid, scoring='accuracy', cv=kfold)
gsearch.fit(X_train_scaled, y_train)
gsearch.best_params_


# In[ ]:


svc = SVC(**gsearch.best_params_)
svc.fit(X_train_scaled, y_train)
svc_preds = svc.predict(X_test_scaled)


# In[ ]:


svc_cm = confusion_matrix(y_test, svc_preds)

sns.heatmap(svc_cm, annot=True)


# In[ ]:


svc_clf_report = classification_report(y_test, svc_preds)
print(svc_clf_report)


# # Tuning and applying MLP Classifier 

# In[ ]:


mlpc = MLPClassifier()

mlpc_grid = {'hidden_layer_sizes' : [(100,), (100, 100), (150, 150), (200, 200)], 
            'alpha' : [0.0001, 0.001, 0.01, 0.1], 
            'learning_rate' : ['constant', 'invscaling', 'adaptive']}

mlpc_kfold = KFold(n_splits=10, shuffle=True)
mlpc_search = GridSearchCV(estimator=mlpc, param_grid=mlpc_grid, scoring='accuracy', cv=mlpc_kfold)
mlpc_search.fit(X_train_scaled, y_train)
mlpc_search.best_params_


# In[ ]:


mlp_clf = MLPClassifier(**mlpc_search.best_params_)
mlp_clf.fit(X_train_scaled, y_train)


# In[ ]:


mlpc_preds = mlp_clf.predict(X_test_scaled)


# In[ ]:


mlpc_cm = confusion_matrix(y_test, mlpc_preds)
sns.heatmap(mlpc_cm, annot=True)


# In[ ]:


mlpc_clf_report = classification_report(y_test, mlpc_preds)
print(mlpc_clf_report)


# # Gaussian Process Classifier 

# In[ ]:


gpc = GaussianProcessClassifier()

gpc_param_grid = {'optimizer' : ['fmin_l_bfgs_b', None], 'n_restarts_optimizer' : [0, 1, 3, 5, 9], 
                 'max_iter_predict' : [50, 100, 200, 500, 1000]}

gpc_kfold = KFold(n_splits=10, shuffle=True)
gpc_grid_search = GridSearchCV(estimator=gpc, param_grid=gpc_param_grid, scoring='accuracy', cv=gpc_kfold)
gpc_grid_search.fit(X_train_scaled, y_train)
print(gpc_grid_search.best_params_)


# In[ ]:


gp_clf = GaussianProcessClassifier(**gpc_grid_search.best_params_)
gp_clf.fit(X_train_scaled, y_train)


# In[ ]:


gpc_preds = gp_clf.predict(X_test_scaled)


# In[ ]:


gpc_cm = confusion_matrix(y_test, gpc_preds)
sns.heatmap(gpc_cm, annot=True)


# In[ ]:


gpc_cr = classification_report(y_test, gpc_preds)
print(gpc_cr)


# # We do not want our soldiers blown up by live mines. So we choose the model with the highest (positive class) recall score

# In[ ]:


print('SVC recall score: {}'.format(recall_score(y_test, svc_preds)))
print('MLPClassifier recall score: {}'.format(recall_score(y_test, mlpc_preds)))
print('GaussianProcessClassifier recall score: {}'.format(recall_score(y_test, gpc_preds)))

