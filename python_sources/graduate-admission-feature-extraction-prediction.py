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


data = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict.csv')


# In[ ]:


data.head()


# In[ ]:


data['University Rating'].value_counts()


# In[ ]:


data['Research'].value_counts()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

for column in data.columns:
    fig_column = plt.hist(data[column])
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()


# In[ ]:


import seaborn as sns

sns.pairplot(data)


# In[ ]:


data.columns


# find the geometric mean of scores to help predictig the probabilities of admission
# 

# In[ ]:


from scipy import stats
 
data['g_mean_score'] = stats.gmean(data.iloc[:,:6],axis=1)


# In[ ]:


data.head()


# In[ ]:


plt.hist(data['g_mean_score'])


# In[ ]:


plt.scatter(data['g_mean_score'], data['Chance of Admit '])
plt.xlabel('g_mean_score')
plt.ylabel('Chance of Admit')
plt.show()


# we can observe that the new feature geometric mean score has a positive correlation with Chance of admit.

# In[ ]:


plt.scatter(data['University Rating'], data['CGPA'])
plt.xlabel('University rating')
plt.ylabel('CGPA')
plt.show()


# In[ ]:


uni_gpa = data[['University Rating', 'CGPA']]


# In[ ]:


Uni_cgpa_avg = uni_gpa.groupby('University Rating').mean().to_dict()


# In[ ]:


uni_cgpa_avg = Uni_cgpa_avg.get('CGPA')


# In[ ]:


uni_cgpa_avg


# In[ ]:


data['avg_cgpa'] = data['University Rating'].map(uni_cgpa_avg)


# In[ ]:


data.head()

creating new column to check if the CGPA is higher or lower than the average for the university rating
# In[ ]:


def f(row):
    if row['CGPA'] > row['avg_cgpa']:
        val = 1
    else:
        val = 0
    return val


# In[ ]:


data['present_chance'] = data.apply(f, axis=1)


# In[ ]:


data.head()


# In[ ]:


X = data.drop(['Chance of Admit ','Serial No.', 'avg_cgpa'], axis = 1)
y = data.iloc[:,8]


# In[ ]:


X.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

ss_scaled_X = ss.fit_transform(X)


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(ss_scaled_X, y, test_size = 0.2, random_state = 42)


# In[ ]:


from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor

models = []
models += [['Ridge', Ridge(alpha = 0.9, solver = "cholesky")]]
models += [['Lasso', Lasso(alpha = 1)]]
models += [['Elastic Net', ElasticNet(alpha = 0.1, l1_ratio = 0.25)]]
models += [['SVM', LinearSVR()]]
models += [['Tree', DecisionTreeRegressor()]]
models += [['Rforest', RandomForestRegressor()]]


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits = 5, random_state = 42)
result_MM =[]
names = []

for name, model in models:
    cv_score = -1 * cross_val_score(model, X_train, y_train, cv = kfold, scoring = 'neg_root_mean_squared_error')
    result_MM +=[cv_score]
    names += [name]
    print('%s: %f (%f)' % (name,cv_score.mean(), cv_score.std()))


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits = 10, random_state = 42)
result_MM =[]
names = []

for name, model in models:
    cv_score = -1 * cross_val_score(model, X_train, y_train, cv = kfold, scoring = 'neg_root_mean_squared_error')
    result_MM +=[cv_score]
    names += [name]
    print('%s: %f (%f)' % (name,cv_score.mean(), cv_score.std()))


# increasing the splits have further increased the RMSE of the ridge regression

# In[ ]:


Ridge_model = Ridge(alpha = 0.9, solver = "cholesky").fit(X_train, y_train)


# In[ ]:


y_preds = Ridge_model.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_error
ridge_rmse = np.sqrt(mean_squared_error(y_test, y_preds))


# In[ ]:


print(f'the RMSE obtained for the Ridge Regression is:{ridge_rmse}.')


# In[ ]:




