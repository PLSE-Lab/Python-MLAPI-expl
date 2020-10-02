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


from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


# In[ ]:


train_df = pd.read_csv("/kaggle/input/predict-the-number-of-upvotes-a-post-will-get/train_NIR5Yl1.csv")
test_df = pd.read_csv("/kaggle/input/predict-the-number-of-upvotes-a-post-will-get/test_8i3B3FC.csv")


# In[ ]:


train_df.head()


# In[ ]:


train_df.isna().sum()


# In[ ]:


train_df.drop(['ID', 'Username'], axis=1, inplace=True)
train_df.head()


# In[ ]:


train_df.info()


# In[ ]:


for label, content in train_df.items() :
    if pd.api.types.is_object_dtype(content) :
        train_df[label] = content.astype('category')


# In[ ]:


train_df.info()


# In[ ]:


for label, content in train_df.items() :
    if pd.api.types.is_categorical_dtype(content) :
        train_df[label] = content.cat.codes+1


# In[ ]:


train_df.info()


# In[ ]:


X = train_df.drop('Upvotes', axis=1)
y = train_df.Upvotes


# In[ ]:


np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[ ]:


models = {"ada_reg" : AdaBoostRegressor(),
          "gra_reg" : GradientBoostingRegressor()}


# In[ ]:


def fit_and_score (models, X_train, y_train, X_test, y_test) :
    models_score = {}
    model_preds = {}
    for name, model in models.items() :
        model.fit(X_train, y_train)
        models_score[name] = model.score(X_test, y_test)
    return models_score


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nmodels_Scores = fit_and_score(models, X_train, y_train, X_test, y_test)\nmodels_Scores')


# In[ ]:


gra_reg =  GradientBoostingRegressor()


# In[ ]:


gra_reg.fit(X_train, y_train)


# In[ ]:


preds = gra_reg.predict(X_test)
mean_absolute_error(y_test, preds)


# In[ ]:


test_df.drop(['ID', 'Username'], axis=1, inplace=True)
test_df.info()


# In[ ]:


for label, content in test_df.items() :
    if pd.api.types.is_object_dtype(content) :
        test_df[label] = content.astype('category')


# In[ ]:


for label, content in test_df.items() :
    if pd.api.types.is_categorical_dtype(content) :
        test_df[label] = content.cat.codes+1


# In[ ]:


test_df.info()


# In[ ]:


test_preds = gra_reg.predict(test_df)
final_result = pd.DataFrame(test_preds)
final_result.head()


# In[ ]:


final_result.columns = ['Upvotes']


# In[ ]:


final_result.head()

