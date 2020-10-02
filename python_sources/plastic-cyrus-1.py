#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from scipy.stats import norm
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


PATH = '../input'
training_set = pd.read_csv(os.path.join(PATH, 'training_set.csv'))
testing_set = pd.read_csv(os.path.join(PATH, 'test_set.csv'), nrows=100)
complete_set = training_set.append(testing_set)
#training_set = training_set.set_index(training_set['object_id'])
#training_set = training_set.drop(columns=['object_id'])


# In[ ]:


FEATURES = list(complete_set)
FEATURES


# In[ ]:


complete_set.head()


# In[ ]:


pd.set_option('display.float_format', lambda x : '%.0f' % x)


# In[ ]:


complete_set.describe()


# In[ ]:


PREDICTORS = ['mjd_a59k', 'passband', 'flux']
OUTCOME = ['detected']


# In[ ]:


def preprocessing(df):
    '''
    return a preprocessed df
    '''
    df_new = df.copy()
    ### SAND BOX
    df_new['mjd_a59k'] = df_new['mjd'] - df_new['mjd'].min()
#     MJD to Unix time conversion: (MJD - 40587) * 86400 + seconds past UTC midnight
#     https://wiki.polaire.nl/doku.php?id=mjd_convert_modified_julian_date
    df_new['unix'] = (df_new['mjd'] - 40587) * 86400
    df_new['unix'] = df_new['unix'] - df_new['unix'].min()
    ### SAND BOX END
    return df_new


# In[ ]:





# In[ ]:


df_current = preprocessing(training_set)
#### SAND BOX ####


# In[ ]:


sns.distplot(df_current[df_current.detected == 0]['flux'] , fit=norm)


# In[ ]:


df_current.describe()


# In[ ]:


# df_current[df_current.detected == 1].describe() - df_current[df_current.detected == 0].describe()


# In[ ]:


df_current[df_current.detected == 0].isna().sum()


# In[ ]:


df_current[df_current.detected == 0].describe()


# In[ ]:


# df_current[df_current.detected == 1].describe()[]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#### SAND BOX ENDED ####


# In[ ]:


def x_y(df):
    '''
    determine the x and y by global variable predictors and outcome
    '''
    X = df[PREDICTORS]
    y = df[OUTCOME]
    return (X, y)


# In[ ]:


training_final = preprocessing(training_set)
testing_final = preprocessing(pd.read_csv(os.path.join(PATH, 'test_set_sample.csv')))


# In[ ]:


X_tr, y_tr = x_y(training_final)
X_te, y_te = x_y(testing_final)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# In[ ]:


models = []
### SAND BOX ####
# models.append(('Logistic Regression', LogisticRegression()))
# models.append(('Extreme Gradient Booster', XGBClassifier()))
models.append(('Extreme Gradient Booster', XGBClassifier(learning_rate=0.05)))
models.append(('Extreme Gradient Booster', XGBClassifier(learning_rate=0.02)))
models.append(('Extreme Gradient Booster', XGBClassifier(learning_rate=0.01)))
models.append(('Extreme Gradient Booster', XGBClassifier(learning_rate=0.2)))
models.append(('Extreme Gradient Booster', XGBClassifier(learning_rate=0.25)))


# In[ ]:


class ModelResult():
    def __init__(self, name, model) :
        self.name = name
        self.model = model
        self.metrics = []
    def set_metric(self, name, metric):
        self.metrics.append((name, metric))
    def __str__(self):
        returner = "###### " + self.name + " ######\n"
        for met_name, metric in self.metrics :
            returner += met_name + ":\n"
            returner += str(metric) + "\n"
        return returner
    


# In[ ]:


# mres = ModelResult('lr', LogisticRegression())
# print(mres)


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
import time


# In[ ]:


def run_models(models):
    results = []
    for name, model in models :
        start = time.time()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        mr = ModelResult(name, model)
        mr.set_metric(('accuracy'), accuracy_score(y_te, y_pred))        
        mr.set_metric(('confusion_matrix'), confusion_matrix(y_te, y_pred))        
        mr.set_metric(('roc_curve'), roc_curve(y_te, y_pred))
        mr.set_metric(('time_elapsed'), time.time() - start)
        results.append(mr)
    return results
#uncomment this if ready
results = run_models(models)


# In[ ]:


for res in results :
    print(res)


# In[ ]:


for res in results :
    print(res)


# In[ ]:


results_3 = run_models(models_3)


# In[ ]:


for r in results_3 :
    print(r)


# PREDICTORS = ['mjd_a59k', 'passband', 'flux']
# 
# ###### Logistic Regression ######
# accuracy:
# 0.948817
# confusion_matrix:
# [[948817      0]
#  [ 51183      0]]
# roc_curve:
# (array([0., 1.]), array([0., 1.]), array([1, 0]))
# 
# ###### Extreme Gradient Booster 0.1 LR ######
# accuracy:
# 0.964415
# confusion_matrix:
# [[948511    306]
#  [ 35279  15904]]
# roc_curve:
# (array([0.00000000e+00, 3.22506869e-04, 1.00000000e+00]), array([0.        , 0.31072817, 1.        ]), array([2, 1, 0]))
# 
# ###### Extreme Gradient Booster 0.05 LR ######
# accuracy:
# 0.962487
# confusion_matrix:
# [[948635    182]
#  [ 37331  13852]]
# roc_curve:
# (array([0.00000000e+00, 1.91817811e-04, 1.00000000e+00]), array([0.        , 0.27063673, 1.        ]), array([2, 1, 0]))
# time_elapsed:
# 49.75889468193054
# 
# ###### Extreme Gradient Booster 0.02 LR ######
# accuracy:
# 0.960895
# confusion_matrix:
# [[948691    126]
#  [ 38979  12204]]
# roc_curve:
# (array([0.00000000e+00, 1.32796946e-04, 1.00000000e+00]), array([0.        , 0.23843854, 1.        ]), array([2, 1, 0]))
# time_elapsed:
# 49.226396322250366
# 
# ###### Extreme Gradient Booster 0.01 LR ######
# accuracy:
# 0.959617
# confusion_matrix:
# [[948680    137]
#  [ 40246  10937]]
# roc_curve:
# (array([0.0000000e+00, 1.4439033e-04, 1.0000000e+00]), array([0.        , 0.21368423, 1.        ]), array([2, 1, 0]))
# time_elapsed:
# 49.153637170791626
# 
# ###### Extreme Gradient Booster 0.20 LR ######
# accuracy:
# 0.965446
# confusion_matrix:
# [[948454    363]
#  [ 34191  16992]]
# roc_curve:
# (array([0.00000000e+00, 3.82581678e-04, 1.00000000e+00]), array([0.        , 0.33198523, 1.        ]), array([2, 1, 0]))
# time_elapsed:
# 49.09350323677063
# 
# ###### Extreme Gradient Booster 0.25 LR######
# accuracy:
# 0.965306
# confusion_matrix:
# [[948430    387]
#  [ 34307  16876]]
# roc_curve:
# (array([0.00000000e+00, 4.07876334e-04, 1.00000000e+00]), array([0.        , 0.32971885, 1.        ]), array([2, 1, 0]))
# time_elapsed:
# 49.166661739349365
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




