#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import csv,pickle,datetime
from sklearn.preprocessing import *
from sklearn.svm import *
from sklearn.utils import resample
from sklearn.pipeline import *
from sklearn.ensemble import *
from sklearn.model_selection import *
import pandas as pd
import numpy as np
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgbm

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


pd.options.display.float_format = '{:.2f}'.format
import matplotlib
matplotlib.pyplot.rcParams['figure.figsize'] = (20, 10)


# In[ ]:


df_train = pd.read_csv("/kaggle/input/train.csv")
df_test = pd.read_csv("/kaggle/input/test.csv")
df_country = pd.read_csv("/kaggle/input/country_info.csv")
df_survay = pd.read_csv("/kaggle/input/survey_dictionary.csv")
df_sample = pd.read_csv("/kaggle/input/sample_submission.csv")


# In[ ]:


df_sample.head(1)


# In[ ]:


df_train.head(1)


# In[ ]:


df_train.columns.values


# In[ ]:


df_train.dtypes


# In[ ]:


df_train.select_dtypes(include=[int,float]).dtypes


# In[ ]:


df_train.select_dtypes(include=[object]).dtypes


# In[ ]:


df_train["ConvertedSalary"].hist(bins=1000)


# In[ ]:


df_train.describe()


# In[ ]:


list(np.arange(0,1,0.1))


# In[ ]:


df_train["ConvertedSalary"].describe(percentiles=list(np.arange(0,1,0.1)))


# In[ ]:


df_str_train = df_train.select_dtypes(include=object)
del df_str_train["MilitaryUS"]
del df_str_train["Student"]
df_str_test = df_test[df_str_train.columns.values]
print(df_str_train.columns.size)
print(df_str_test.columns.size)
#df_str_train.head(1).values


# In[ ]:


print(len(df_str_train))
print(len(df_str_test))
df_all = pd.concat([df_str_train,df_str_test]).fillna("NA------")
df_all = df_all.apply(LabelEncoder().fit_transform)
df_le_train = df_all[:42893]
df_le_test = df_all[42893:]
print(len(df_le_train))
print(len(df_le_test))


# In[ ]:


df_num_train = df_train.select_dtypes(include=[int,float]).fillna(0)
df_num_test  = df_test.select_dtypes(include=[int,float]).fillna(0)


# In[ ]:


df_show=df_num_train[[
        "ConvertedSalary",
        "AssessBenefits1",
        "AssessBenefits2",
        "AssessBenefits3",
        "AssessBenefits4",
        "AssessBenefits5",

    ]].sample(frac=0.25)
plot = sns.pairplot(
    data = df_show
)


# In[ ]:


df_show.head(10)


# In[ ]:


df_num_train["ConvertedSalary"].describe(percentiles=list(np.arange(0,1,0.1)))


# In[ ]:


X_train = df_num_train[df_num_train.columns[df_num_train.columns != "ConvertedSalary"]]
X_train = df_num_train[df_num_train.columns[df_num_train.columns != "Respondent"]]
y_train = df_num_train["ConvertedSalary"].values
#X_train = np.c_[X_train,df_le_train.values]
X_train = df_le_train.values
print(X_train.shape)
print(y_train.shape)


# In[ ]:


X_test = df_num_test[df_num_test.columns[df_num_test.columns != "MilitaryUS"]]
X_test = df_num_test[df_num_test.columns[df_num_test.columns != "Respondent"]]
X_test = X_test.values
#X_test = np.c_[X_test,df_le_test.values]
X_test = df_le_test.values
print(X_test.shape)


# In[ ]:


import sklearn
sorted(sklearn.metrics.SCORERS.keys())


# In[ ]:


from sklearn.linear_model import *
model = lgbm.LGBMRegressor()
#model = LinearRegression()

#model = LinearSVR()

param_grid={
#    "n_estimators" : [50,100,200,300,500],
#    "max_depth":[1,3,5,7]
}

#RMSLE Scoring
from sklearn.metrics import make_scorer
def rmsle(predicted, actual, size):
    return np.sqrt(np.nansum(np.square(np.log(predicted + 1) - np.log(actual + 1)))/float(size))
scorer = make_scorer(rmsle, greater_is_better=False, size=10)

#Grid Search
grid = GridSearchCV(
    model,
    param_grid,
    scoring=scorer,
    #    scoring="neg_mean_squared_log_error",
    cv=2,
    verbose=2,
    #n_jobs=2
)

grid.fit(
    X_train,
    y_train
)

pred = grid.predict(X_test)
best = grid.best_estimator_
best.fit(X_train,y_train)
pred = best.predict(X_test)


# In[ ]:


def displayResult(grid):
    means = np.array(grid.cv_results_['mean_test_score'])
    stds = grid.cv_results_['std_test_score']
    fittime = grid.cv_results_['mean_fit_time']

    #    maxidx = np.where(means == np.max(means))[0][0]

    for mean, std, params ,time,i in zip(means, stds, grid.cv_results_['params'],fittime,range(100000)):
        print("%1.6f (+/-%1.6f) %4.2f for %r" % (
            mean,
            std,
            time,
            params
        ))
    #        if i-1 == maxidx: print("* ",end='')

    #print("---------")
    #print(grid.best_estimator_.named_steps["select"])
    print("---------")
    print(grid.best_estimator_)
    print("---------")
    print(grid.best_score_)

displayResult(grid)


# In[ ]:


grid.best_estimator_.feature_importances_


# In[ ]:


ids = df_test["Respondent"].values
submit = np.c_[ids,pred]

df_sub = pd.DataFrame(
    data=submit,
    columns=[
        "Respondent",
        "ConvertedSalary"
    ]
)

df_sub["Respondent"] = df_sub["Respondent"].astype(int)
df_sub = df_sub.set_index("Respondent")
df_sub.head(10)


# In[ ]:


df_sub["ConvertedSalary"].describe(percentiles=list(np.arange(0,1,.1)))


# In[ ]:


df_sub[df_sub.ConvertedSalary < 0] = 7200


# In[ ]:


df_sub.head(10)


# In[ ]:


df_sub["ConvertedSalary"].describe(percentiles=list(np.arange(0,1,.1)))


# In[ ]:


df_train["ConvertedSalary"].describe(percentiles=list(np.arange(0,1,.1)))


# In[ ]:


df_sub["ConvertedSalary"].hist()
df_sub.head(100)


# In[ ]:


df_sub.to_csv(
    "out.csv",
    columns=[
         "ConvertedSalary"
    ],
    index=True
)

