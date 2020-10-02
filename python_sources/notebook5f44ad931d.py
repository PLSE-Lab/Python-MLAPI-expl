#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/train.csv')


# In[ ]:


df.head()


# In[ ]:


df.type.unique()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.manifold import TSNE
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


col_n1 = df.columns[1:5]
y_n = ['type']
print(col_n1)


# In[ ]:


df = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df.fillna(-1, inplace=True)
df_test.fillna(-1, inplace=True)


# In[ ]:


sns.pairplot(df, hue='type')


# In[ ]:


lab = LabelEncoder()
df['col_lab'] = lab.fit_transform(df['color'])
ff = PolynomialFeatures(degree=2)
a = ff.fit_transform(df[col_n1])
df_t = pd.DataFrame(a)
df = pd.concat([df,df_t],axis=1)
df.head()


# In[ ]:


col_n=[]
for a in df.columns:
    if a not in ['id', 'color', 'type']:
        col_n.append(a)


# In[ ]:


print(col_n)
ts = TSNE()
m = ts.fit_transform(df[col_n])
df_t = pd.DataFrame(m)
df_t.columns = ['x', 'y']
df = pd.concat([df,df_t],axis=1)


# In[ ]:


df['c'] = 'green'
df.ix[df['type'] == 'Ghoul', 'c'] = 'red'
df.ix[df['type'] == 'Ghost', 'c'] = 'blue'
plt.scatter(df['x'], df['y'], c = df['c'])


# In[ ]:


X_tr, X_te, y_tr, y_te = train_test_split(df[col_n], df[y_n])


# In[ ]:


clf = RandomForestClassifier(n_estimators=150)


# In[ ]:


clf.fit(X_tr, y_tr)


# In[ ]:


pred = clf.predict(X_te)


# In[ ]:


accuracy_score(y_pred=pred, y_true=y_te)


# In[ ]:


svm = SVC(C=20, decision_function_shape='ovr')


# In[ ]:



svm.fit(X_tr, y_tr)


# In[ ]:


pred2 = svm.predict(X_te)


# In[ ]:


accuracy_score(y_pred=pred2, y_true=y_te)


# In[ ]:


labels = df['type']
scores_f = cross_val_score(clf, df[col_n], labels, cv=10, n_jobs=-1)
scores_s = cross_val_score(svm, df[col_n], labels, cv=10, n_jobs=-1)
print(scores_f.mean())
print(scores_f)
print(scores_s.mean())
print(scores_s)


# In[ ]:


ff = PolynomialFeatures()
a = ff.fit_transform(df[col_n1])
df_t = pd.DataFrame(a)
df_test = pd.concat([df_test,df_t],axis=1)
print(col_n)
print(df_test.head())


# In[ ]:


df_test['type'] = clf.predict(df_test[col_n])


# In[ ]:


df_test.dtypes


# In[ ]:


df.dtypes


# In[ ]:


df_test.fillna(-1, inplace=True)


# In[ ]:


np.any(np.isnan(df_test[col_n]))


# In[ ]:


df_test[['id', 'type']].to_csv('test.csv')


# In[ ]:


print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:




