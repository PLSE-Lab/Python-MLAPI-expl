#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import catboost
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
import seaborn as sns
import shap
from scipy.constants import golden
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
shap.initjs()
np.random.seed(42)


# In[ ]:


df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')


# In[ ]:


print(df.head()); print(df.columns)


# In[ ]:


df.head()


# In[ ]:


# left skewed distribution. raise your standard university!


# In[ ]:


sns.distplot(df['Chance of Admit '], hist=False)


# In[ ]:


y = df['Chance of Admit ']
df = df.drop('Chance of Admit ', axis=1)


# no missing values

# In[ ]:


df.describe(); df.isnull().sum()


# all numeric features

# In[ ]:


df.dtypes


# In[ ]:


m = CatBoostRegressor(iterations=1000, silent=True)


# In[ ]:


valid_idx = 400


# In[ ]:


X_train, y_train, X_valid, y_valid = df[:valid_idx], y[:valid_idx], df[valid_idx:], y[valid_idx:]


# In[ ]:


train_pool, validate_pool = Pool(X_train, y_train), Pool(X_valid, y_valid)


# In[ ]:


m.fit(train_pool, eval_set=validate_pool, plot=True, logging_level='Silent')


# In[ ]:


# create series feature importance
imp = pd.Series(m.feature_importances_, index=df.columns).sort_values()


# In[ ]:


# apparently university rating is pretty unimportant. You'd expect the best universities to have better admission rates.


# In[ ]:


plt.figure(figsize=(10, 10/golden), dpi=144)
sns.barplot(imp, imp.index)


# That's surprising. You'd expect the best universities to have worse admission rates. Let's see if that's the case.

# In[ ]:


plot = pd.concat([X_train, y], axis=1)
sns.lineplot('University Rating', 'Chance of Admit ', data=plot)


# Hmm. It it's not the case. The best universities admit the most. Lol. Maybe it's because they're the largest. Anyway, the fact that it isn't an important variable must mean that there are other variables picking up its signal.

# In[ ]:


sns.heatmap(X_train.corr())


# University rating is highly correlated with other variables in the dataset. So, let's drop it.

# In[ ]:


m = CatBoostRegressor(iterations=1000, silent=True, ignored_features=['University Rating'])
m.fit(train_pool, eval_set=validate_pool, plot=True)


# That improved validation rmse amazingly. Hooray.

# In[ ]:


def feature_importance(model):
    """Return feature importance of catboost model and plot it."""
    imp = pd.Series(model.feature_importances_, index=df.columns).sort_values()
    plt.figure(figsize=(10, 10 / golden), dpi=144)
    sns.barplot(imp, imp.index)
    return imp


# In[ ]:


imp = feature_importance(m)


# didn't change feature importance a huge amount. some features traded places. However, let's keep it dropped so we can interpret the results, we can add it back later for slightly better predictive power.

# In[ ]:


explainer = shap.TreeExplainer(m)
shap_values = explainer.shap_values(X_train)


# trained shap model to interpret the model

# In[ ]:


shap.force_plot(explainer.expected_value, shap_values[2,:], X_train.iloc[2,:])


# CGPA had a big effect on the first sample. serial no. had some impact. You would expect that serial number would be unpredictive. It must be proxying some sort of hidden variable. Perhaps date?

# In[ ]:


plt.axvline(8, 0, 0.68, color='red')
sns.distplot(plot['CGPA'], hist=False)


# 8 is a below average gpa which makes sense that it would have a negative impact on the student's predicted admission.

# In[ ]:


sns.regplot('CGPA', 'Chance of Admit ', data=plot,lowess=True)


# Linear relationship between CGPA and admission. Not surprising!

# In[ ]:


sns.regplot('Serial No.', 'Chance of Admit ', data=plot, lowess=True)


# It doesn't look too related with admission. however, it's weird there's a lump in the middle. There's probably a latent variable involved. Maybe low serial numbers are people who applied first, and later serial
# numbers people who applied late.

# In[ ]:


m = CatBoostRegressor(iterations=1000, silent=True, ignored_features=['University Rating', 'Serial No.'])
m.fit(train_pool, eval_set=validate_pool, plot=True)


# Dropping serial no. increased validation rmse.

# In[ ]:


imp = feature_importance(m)


# It didn't change order of feature importance. I say include it and let it remain a mystery. Lol.

# let's add back all features

# In[ ]:


m = CatBoostRegressor(iterations=1000, silent=True)
m.fit(train_pool, eval_set=validate_pool, plot=True)


# In[ ]:


shap.dependence_plot("Serial No.", shap_values, X_train, interaction_index=None)


# Hmmm. Lower serial nos have a big negative impact on admission. Serial no. is definitely not random.

# In[ ]:


shap.dependence_plot("CGPA", shap_values, X_train, interaction_index=None)


# Almost linear relationship with CGP

# In[ ]:


shap.summary_plot(shap_values, X_train)


# * Unsuprisingly CGPA shows the largest variance in model impact. High values increase admission rate and low values decrease admission rate.
# * Seems low serial numbers have a chance of lower admission rate. Maybe it's people applying early and to a lot of universities because their expectation of getting in is low. Worth investigating whover put this data together.
# * TOEFL score is proportional to admission rate. If you have research experience you're slightly more likely to get it.
# * Having a good SOP helps, and a letter of recommentation helps.
# * All else equal, university rating doesn't impact admission rate so I will exclude it from final model. This explains why dropping it slightly improved validation metric.

# In[ ]:


m = CatBoostRegressor(iterations=1000, silent=True, ignored_features=['University Rating'])
m.fit(train_pool, eval_set=validate_pool, plot=True)


# In[ ]:


explainer.shap_interaction_values(X_train)


# In[ ]:


y_pred = m.predict(X_valid)
round(mean_squared_error(y_valid, y_pred, squared=False), 4)


# # TO-DO
# * tune model
# * investigate interactions
# * figure out what is going on with serial no.
# * save model without university rating instead of training it again

# In[ ]:


shap.__version__


# In[ ]:




