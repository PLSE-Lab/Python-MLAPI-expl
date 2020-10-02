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


# Imported Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


#read data
data = pd.read_csv('/kaggle/input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv')
data.head()


# In[ ]:


data.shape


# In[ ]:


data.isnull().sum()


# In[ ]:


data.dtypes


# In[ ]:


data = data.drop(columns='ID')


# In[ ]:


print('No', round(data['default.payment.next.month'].value_counts()[0]/len(data) * 100,2), '% of the dataset')
print('Yes', round(data['default.payment.next.month'].value_counts()[1]/len(data) * 100,2), '% of the dataset')


# In[ ]:


colors = ["#0101DF", "#DF0101"]

sns.countplot('default.payment.next.month', data=data, palette=colors)
plt.title('Class Distributions \n (0: No || 1: Yes)', fontsize=14)


# In[ ]:


X = data.iloc[:, data.columns != 'default.payment.next.month']
y = data.iloc[:, data.columns == 'default.payment.next.month']


# In[ ]:


corr_matrix = X.corr().abs()
mask = np.array(corr_matrix)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(30,30)
sns.heatmap(corr_matrix, mask=mask,vmax=.5, square=True,annot=True)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[ ]:


from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10, random_state=42)


# In[ ]:


from sklearn.model_selection import cross_val_score
model1 = XGBClassifier()
results = cross_val_score(model1, X_train, y_train, cv=kfold)

cv_mean = results.mean()
cv_std = results.std()
print("Accuracy: %.2f%% (%.2f%%)" % (cv_mean,cv_std ))


# In[ ]:


model1 = XGBClassifier()
model1.fit(X_train, y_train)
print(roc_auc_score(y_test, model1.predict(X_test)))
print(accuracy_score(y_test, model1.predict(X_test)))


# In[ ]:


importances = model1.feature_importances_
indices = np.argsort(importances)[::-1]
names = [X.columns[i] for i in indices]
plt.figure(figsize=(30,15))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), names, rotation=90)
plt.show()


# In[ ]:


model2 = CatBoostClassifier(logging_level='Silent')
results = cross_val_score(model2, X_train, y_train, cv=kfold)

cv_mean = results.mean()
cv_std = results.std()
print("Accuracy: %.2f%% (%.2f%%)" % (cv_mean,cv_std ))


# In[ ]:


model2 = CatBoostClassifier(logging_level='Silent')
model2.fit(X_train, y_train)
print(roc_auc_score(y_test, model2.predict(X_test)))
print(accuracy_score(y_test, model2.predict(X_test)))


# In[ ]:


importances = model2.feature_importances_
indices = np.argsort(importances)[::-1]
names = [X.columns[i] for i in indices]
plt.figure(figsize=(30,15))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), names, rotation=90)
plt.show()


# In[ ]:


data['SEX'] = data['SEX'].astype('category')
data['EDUCATION'] = data['EDUCATION'].astype('category')
data['MARRIAGE'] = data['MARRIAGE'].astype('category')
data['PAY_0'] = data['PAY_0'].astype('category')
data['PAY_2'] = data['PAY_2'].astype('category')
data['PAY_3'] = data['PAY_3'].astype('category')
data['PAY_4'] = data['PAY_4'].astype('category')
data['PAY_5'] = data['PAY_5'].astype('category')
data['PAY_6'] = data['PAY_6'].astype('category')


# In[ ]:


data1 = pd.get_dummies(data, drop_first=True)


# In[ ]:


data1.shape


# In[ ]:


X1 = data1.iloc[:, data1.columns != 'default.payment.next.month']
y1 = data1.iloc[:, data1.columns == 'default.payment.next.month']


# In[ ]:


X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.25, random_state=42)


# In[ ]:


model3 = XGBClassifier()
results = cross_val_score(model3, X1_train, y1_train, cv=kfold)

cv_mean = results.mean()
cv_std = results.std()
print("Accuracy: %.2f%% (%.2f%%)" % (cv_mean,cv_std ))


# In[ ]:


model3 = XGBClassifier()
model3.fit(X1_train, y1_train)
print(roc_auc_score(y1_test, model3.predict(X1_test)))
print(accuracy_score(y1_test, model3.predict(X1_test)))


# In[ ]:


importances = model3.feature_importances_
indices = np.argsort(importances)[::-1]
names = [X1.columns[i] for i in indices]
plt.figure(figsize=(30,15))
plt.title("Feature Importance")
plt.bar(range(X1.shape[1]), importances[indices])
plt.xticks(range(X1.shape[1]), names, rotation=90)
plt.show()


# In[ ]:


model4 = CatBoostClassifier(logging_level='Silent')
results = cross_val_score(model4, X1_train, y1_train, cv=kfold)

cv_mean = results.mean()
cv_std = results.std()
print("Accuracy: %.2f%% (%.2f%%)" % (cv_mean,cv_std ))


# In[ ]:


model4 = CatBoostClassifier(logging_level='Silent')
model4.fit(X1_train, y1_train)
print(roc_auc_score(y1_test, model4.predict(X1_test)))
print(accuracy_score(y1_test, model4.predict(X1_test)))


# In[ ]:


importances = model4.feature_importances_
indices = np.argsort(importances)[::-1]
names = [X1.columns[i] for i in indices]
plt.figure(figsize=(30,15))
plt.title("Feature Importance")
plt.bar(range(X1.shape[1]), importances[indices])
plt.xticks(range(X1.shape[1]), names, rotation=90)
plt.show()


# In[ ]:


X_pay = data1.loc[:, 'PAY_0_-1':]
X_pay


# In[ ]:


Xp_train, Xp_test, yp_train, yp_test = train_test_split(X_pay, y1, test_size=0.25, random_state=42)


# In[ ]:


model5 = XGBClassifier()
results = cross_val_score(model5, Xp_train, yp_train, cv=kfold)

cv_mean = results.mean()
cv_std = results.std()
print("Accuracy: %.2f%% (%.2f%%)" % (cv_mean,cv_std ))


# In[ ]:


model5 = XGBClassifier()
model5.fit(Xp_train, yp_train)
print(roc_auc_score(yp_test, model5.predict(Xp_test)))
print(accuracy_score(yp_test, model5.predict(Xp_test)))


# In[ ]:


model6 = CatBoostClassifier(logging_level='Silent')
results = cross_val_score(model6, Xp_train, yp_train, cv=kfold)

cv_mean = results.mean()
cv_std = results.std()
print("Accuracy: %.2f%% (%.2f%%)" % (cv_mean,cv_std ))


# In[ ]:


model6 = CatBoostClassifier(logging_level='Silent')
model6.fit(Xp_train, yp_train)
print(roc_auc_score(yp_test, model6.predict(Xp_test)))
print(accuracy_score(yp_test, model6.predict(Xp_test)))


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

model7 = XGBClassifier(eval_metric='auc')
parameters = {'max_depth'     : [5,6,7,8,9],
              'learning_rate' : [0.01,0.1, 0.2, 0.3],
              'n_estimators'  : [500, 1000, 1500],
              'subsample'     : [0.7, 0.8, 0.9]
             }
random_xb = RandomizedSearchCV(estimator=model7, param_distributions = parameters, cv = 10, n_jobs=-1, random_state=42)
random_xb.fit(X_train, y_train)
print(random_xb.best_params_)


# In[ ]:


print(roc_auc_score(y_test, random_xb.predict(X_test)))
print(accuracy_score(y_test, random_xb.predict(X_test)))


# In[ ]:


model8 = CatBoostClassifier(logging_level='Silent', eval_metric='AUC')
parameters = {'depth'         : [5,6,7,8,9],
              'learning_rate' : [0.01,0.1, 0.2, 0.3],
              'n_estimators'  : [500, 1000, 1500],
             }
random_cb = RandomizedSearchCV(estimator=model8, param_distributions = parameters, cv = 10, n_jobs=-1, random_state=42)
random_cb.fit(X_train, y_train)
print(random_cb.best_params_)


# In[ ]:


print(roc_auc_score(y_test, random_cb.predict(X_test)))
print(accuracy_score(y_test, random_cb.predict(X_test)))


# In[ ]:




