#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


DATA_PATH = '/kaggle/input/mushroom-classification/'


# In[ ]:


file_path = os.path.join(DATA_PATH,'mushrooms.csv')


# In[ ]:


pd.set_option('display.max_columns',30)


# In[ ]:


df = pd.read_csv(file_path)


# In[ ]:


print(f'shape of csv file: {df.shape}')


# In[ ]:


df.head()


# In[ ]:


df.columns = ['target', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
       'ring-type', 'spore-print-color', 'population', 'habitat']


# In[ ]:


for i in df.columns:
    print(f'{i} -> {df[i].unique()}')


# # Exploratory Data Analysis

# In[ ]:


for i in df.columns:
    if df[i].dtype == 'object':
        df[i] = pd.factorize(df[i])[0]


# In[ ]:


df.groupby(['cap-shape'])['target'].value_counts()


# In[ ]:


pd.crosstab(df['cap-shape'],df['target'])


# In[ ]:


fig = px.violin(df,
          x = df['cap-shape'],
          y=df['target'])
fig.show()


# In[ ]:


fig = px.violin(df,
          x = df['cap-surface'],
          y=df['target'])
fig.show()


# #### Feature selection

# In[ ]:


from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.feature_selection import mutual_info_classif


# In[ ]:


y = df.target


# In[ ]:


df.drop('target',axis =1,inplace=True)


# In[ ]:


x = df


# In[ ]:


vrt = VarianceThreshold(threshold=0.01)
vrt.fit(x,y)


# In[ ]:


sum(vrt.get_support())


# In[ ]:


X = vrt.transform(df)


# In[ ]:


chi2_selector = SelectKBest(chi2, k=11)
X_kbest = chi2_selector.fit_transform(X, y)


# In[ ]:


X_kbest.shape


# In[ ]:


mut_feat = mutual_info_classif(X_kbest,y)


# In[ ]:


mut_feat


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X_kbest,y,test_size=0.15,random_state=1)


# In[ ]:


lr = LogisticRegression(max_iter=200)
lr.fit(X_train,y_train)


# In[ ]:


lr.score(X_train,y_train)


# In[ ]:


cross_val_score(lr,X_train,y_train,cv=5)


# In[ ]:


lr.score(X_test,y_test)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf = RandomForestClassifier(max_features=9,max_depth=5,n_estimators=10)


# In[ ]:


rf.fit(X_train,y_train)


# In[ ]:


rf.score(X_train,y_train)


# In[ ]:


cross_val_score(rf,X_train,y_train,cv=5)


# In[ ]:


rf.feature_importances_


# In[ ]:


rf.score(X_test,y_test)


# In[ ]:


from sklearn.metrics import classification_report,roc_auc_score,roc_curve,auc


# In[ ]:


y_pred = rf.predict(X_test)


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


roc_auc_score(y_test,y_pred)


# In[ ]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred)


# In[ ]:


plt.plot(fpr,tpr)
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title(f'tpr vs fpr plot with auc: {roc_auc_score(y_test,y_pred)}')
plt.show()


# In[ ]:




