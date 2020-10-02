#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

#from sklearn.inspection import display_estimator
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, Normalizer
from sklearn.impute import SimpleImputer


# In[ ]:


plt.style.use('fivethirtyeight')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 8


# In[ ]:


admission_data = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
admission_data.drop(['Serial No.'], inplace=True, axis=1)
admission_data.head()


# In[ ]:


# No null data
admission_data.isna().any()


# In[ ]:


sns.distplot(admission_data['Chance of Admit '])
plt.axhline()
plt.axvline(linewidth=0.5)
plt.show()


# In[ ]:


pd.cut(admission_data['Chance of Admit '], bins=5).value_counts()


# In[ ]:


# Higher the chance better it is!
admission_data['admit'] = admission_data['Chance of Admit '] > .75


# In[ ]:


admission_data.head()


# In[ ]:


# Seems like we have linear relation here. Univery 4 & 5 seem to be very picky but exceptions do exists
sns.pairplot(data=admission_data, y_vars='Chance of Admit ', x_vars=['GRE Score', 'TOEFL Score', 'CGPA'], hue='University Rating', height=4);


# In[ ]:


fig, axes = plt.subplots(1, 3, figsize=(15, 3))
sns.boxplot(data=admission_data, y='Chance of Admit ', x='University Rating', ax=axes[0])
sns.boxplot(data=admission_data, y='Chance of Admit ', x='SOP', ax=axes[1])
sns.boxplot(data=admission_data, y='Chance of Admit ', x='LOR ', ax=axes[2])
fig.tight_layout()


# In[ ]:


sns.heatmap(admission_data.corr(), annot=True, fmt=".2f")


# Your change of getting admitted boils down to **GRE Score, TOEFL Score and CGPA**

# ## Model

# In[ ]:


selected_numeric_columns = ["GRE Score", "TOEFL Score", "CGPA", 'admit']


# In[ ]:


admission_data = admission_data[selected_numeric_columns]


# In[ ]:


admission_data_train = admission_data[:400]
admission_data_test = admission_data[400:]


# In[ ]:


admission_data_train_X = admission_data_train.drop(['admit'], axis=1)
admission_data_train_y = admission_data_train['admit']

admission_data_test_X = admission_data_test.drop(['admit'], axis=1)
admission_data_test_y = admission_data_test['admit']


# In[ ]:


pipe = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(min_samples_leaf = 5, n_estimators = 100))
])


# In[ ]:


pipe = pipe.fit(admission_data_train_X, admission_data_train_y)


# In[ ]:


y_hat = pipe.predict(admission_data_test_X)


# In[ ]:


roc_auc_score(admission_data_test_y, y_hat)


# In[ ]:





# In[ ]:




