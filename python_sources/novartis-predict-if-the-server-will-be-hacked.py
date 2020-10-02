#!/usr/bin/env python
# coding: utf-8

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


import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import f1_score


# In[ ]:


train = pd.read_csv("/kaggle/input/novartis-data/Train.csv")
submit = pd.read_csv("/kaggle/input/novartis-data/sample_submission.csv")
test = pd.read_csv("/kaggle/input/novartis-data/Test.csv")


# In[ ]:


train.head()


# In[ ]:


sns.countplot(train.MULTIPLE_OFFENSE.value_counts())


# In[ ]:


train.MULTIPLE_OFFENSE.value_counts()


# In[ ]:


test.shape


# In[ ]:



train.head(2)


# In[ ]:


X = train.drop(['MULTIPLE_OFFENSE', 'DATE', 'INCIDENT_ID'],axis=1)
eval_X = test.drop(['DATE','INCIDENT_ID'],axis=1)
Y = train['MULTIPLE_OFFENSE']

incident_ids_train = train['INCIDENT_ID']
incdent_ids_test = test['INCIDENT_ID']


# In[ ]:


#splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33)
X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.33)


# In[ ]:


X_train.shape, y_train.shape,X_test.shape


# In[ ]:


X_train.fillna(0, inplace=True)
X_cv.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)
eval_X.fillna(0, inplace=True)


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = pd.DataFrame(scaler.transform(X_train), columns = X_train.columns)


# In[ ]:


X_cv = pd.DataFrame(scaler.transform(X_cv), columns = X_cv.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)
eval_X = pd.DataFrame(scaler.transform(eval_X), columns = eval_X.columns)


# # **Applying GradientBoostingClassifier**

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
#Additional scklearn functions
from sklearn.model_selection import GridSearchCV


# In[ ]:


X_train.shape, y_train.shape


# In[ ]:


X_train.head()


# # **Hyper parameter tuning**

# In[ ]:


#Choose all predictors except target & IDcols
predictors = [x for x in X_train.columns]
param_test1 = {'n_estimators':range(140,401,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10), 
param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(X_train[predictors],y_train)


# In[ ]:


gsearch1.best_params_, gsearch1.best_score_


# In[ ]:


param_test2 = {'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=320, max_features='sqrt', subsample=0.8, random_state=10), 
param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2.fit(X_train[predictors],y_train)
gsearch2.best_params_, gsearch2.best_score_


# In[ ]:


param_test4 = {'max_features':range(7,20,2)}
gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=320,max_depth=7, min_samples_split=200, min_samples_leaf=60, subsample=0.9, random_state=10),
param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch4.fit(X_train[predictors],y_train)
gsearch4.best_params_, gsearch4.best_score_


# In[ ]:


param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=320,max_depth=7,min_samples_split=200, min_samples_leaf=60, subsample=0.8, random_state=10,max_features=7),
param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch5.fit(X_train[predictors],y_train)
gsearch5.best_params_, gsearch5.best_score_


# In[ ]:


gbm_tuned_1 = GradientBoostingClassifier(learning_rate=0.05, n_estimators=320,max_depth=7, min_samples_split=200,min_samples_leaf=40, subsample=0.90, random_state=10, max_features=7)
gbm_tuned_1


# In[ ]:


gbm_tuned_1.fit(X_train, y_train)


# In[ ]:


res = gbm_tuned_1.predict(X_test)


# In[ ]:


print(f1_score(y_test, res))


# In[ ]:


res = gbm_tuned_1.predict(eval_X)
res_df = pd.DataFrame({'MULTIPLE_OFFENSE':res, 'INCIDENT_ID': incdent_ids_test})
res_df1 = res_df[['INCIDENT_ID','MULTIPLE_OFFENSE']]


# In[ ]:


res_df.shape, res_df1.shape


# In[ ]:


res_df1.to_csv("output_2.csv",index = False)


# # Trying out with Knn, GaussianNB, SVC

# In[ ]:


from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

best_k_value = 13

neigh = svm.SVC(probability=True, class_weight={0: 10})



# neigh = KNeighborsClassifier(n_neighbors=best_k_value)

# neigh = GaussianNB()
neigh.fit(X_train, y_train)

train_fpr, train_tpr, thresholds = roc_curve(y_train, neigh.predict_proba(X_train)[:,1])
test_fpr, test_tpr, thresholds = roc_curve(y_test, neigh.predict_proba(X_test)[:,1])

plt.plot(train_fpr, train_tpr, label = 'TRAIN')
plt.plot(test_fpr, test_tpr, label = 'TEST')
plt.legend()
plt.xlabel('K')
plt.ylabel('AUC')
plt.title('Error Plots')
plt.show()


# In[ ]:


from sklearn.metrics import confusion_matrix 
from sklearn.metrics import f1_score

y_train_predict = neigh.predict(X_train)
y_test_predict = neigh.predict(X_test)

train_confusion_matrix = confusion_matrix(y_train, y_train_predict)
test_confusion_matrix = confusion_matrix(y_test, y_test_predict)


# In[ ]:


print("train CM:")
# print(train_confusion_matrix)

print("test CM:")
print(test_confusion_matrix)


# In[ ]:


print("Training F1 score")
print(f1_score(y_train, y_train_predict))
print("Test F1 score")
print(f1_score(y_test, y_test_predict))


# In[ ]:


# eval_X.head()


# In[ ]:


res = neigh.predict(eval_X)
res_df = pd.DataFrame({'MULTIPLE_OFFENSE':res, 'INCIDENT_ID': incdent_ids_test})
res_df1 = res_df[['INCIDENT_ID','MULTIPLE_OFFENSE']]


# In[ ]:





# In[ ]:


print(res_df.shape)


# In[ ]:


res_df1.head()


# In[ ]:


res_df1.to_csv("results3.csv",index = False)


# In[ ]:





# In[ ]:




