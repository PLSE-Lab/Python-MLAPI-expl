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


dframe = pd.read_csv('../input/mushrooms.csv')


# In[ ]:


dframe.head()


# In[ ]:


y = dframe['class']
X = dframe.drop('class',axis=1)


# In[ ]:


X.columns


# In[ ]:


X.info()


# In[ ]:


import seaborn as sns
import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# **DATA VISUALIZATION [UNIVARIATE ANALYSIS]**

# In[ ]:


sns.countplot(x='cap-shape',data=dframe)


# In[ ]:


sns.countplot(x='cap-surface',data=dframe)


# In[ ]:


sns.countplot(x='cap-color',data=dframe)


# In[ ]:


sns.countplot(x='bruises',data=dframe)


# In[ ]:


sns.countplot(x='odor', data=dframe)


# In[ ]:


sns.countplot(x='gill-attachment',data=dframe)


# In[ ]:


sns.countplot(x='gill-spacing',data=dframe)


# In[ ]:


sns.countplot(x='gill-size',data=dframe)


# In[ ]:


sns.countplot(x='gill-color',data=dframe)


# In[ ]:


sns.countplot(x='stalk-shape',data=dframe)


# In[ ]:


sns.countplot(x='stalk-root',data=dframe)


# In[ ]:


sns.countplot(x='stalk-surface-above-ring',data=dframe)


# In[ ]:


sns.countplot(x='stalk-surface-below-ring',data=dframe)


# In[ ]:


sns.countplot(x='stalk-color-above-ring',data=dframe)


# In[ ]:


sns.countplot(x='stalk-color-below-ring',data=dframe)


# In[ ]:


sns.countplot(x='veil-type',data=dframe)


# In[ ]:


sns.countplot(x='veil-color',data=dframe)


# In[ ]:


sns.countplot(x='ring-number',data=dframe)


# In[ ]:


sns.countplot(x='ring-type',data=dframe)


# In[ ]:


sns.countplot(x='spore-print-color',data=dframe)


# In[ ]:


sns.countplot(x='population',data=dframe)


# In[ ]:


sns.countplot(x='habitat',data=dframe)


# **Dependent variable countplot**

# In[ ]:


sns.countplot(x='class',data=dframe)


# **DATA CLEANING**

# In[ ]:


dd = dframe[dframe['stalk-root']=='?']


# In[ ]:


dd['stalk-root'].value_counts()  #values containing '?' i.e. missing values in stalk root


# In[ ]:


dframe['class'].value_counts().sum()


# In[ ]:


dframe = dframe[dframe['stalk-root'] != '?'] #creating new dataframe without missing walues because they cannot be replaceable.


# In[ ]:


dframe.info()


# **BIVARIATE ANALYSIS**

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[ ]:


dframe['class'] = le.fit_transform(dframe['class']) #label encoding


# In[ ]:


dframe.head()


# In[ ]:


y = pd.DataFrame(dframe['class'],columns=['class'])


# In[ ]:


y.head()


# In[ ]:


X = dframe.drop('class',axis=1,inplace=True)


# In[ ]:


X = dframe


# In[ ]:


X.head()


# In[ ]:


y.head()


# In[ ]:


X_enc=pd.get_dummies(X)


# In[ ]:


X_enc.columns


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


sc = StandardScaler()


# In[ ]:


X_std = sc.fit_transform(X_enc)


# In[ ]:


X = pd.DataFrame(X_std,columns=['cap-shape_b', 'cap-shape_c', 'cap-shape_f', 'cap-shape_k',
       'cap-shape_s', 'cap-shape_x', 'cap-surface_f', 'cap-surface_g',
       'cap-surface_s', 'cap-surface_y', 'cap-color_b', 'cap-color_c',
       'cap-color_e', 'cap-color_g', 'cap-color_n', 'cap-color_p',
       'cap-color_w', 'cap-color_y', 'bruises_f', 'bruises_t', 'odor_a',
       'odor_c', 'odor_f', 'odor_l', 'odor_m', 'odor_n', 'odor_p',
       'gill-attachment_a', 'gill-attachment_f', 'gill-spacing_c',
       'gill-spacing_w', 'gill-size_b', 'gill-size_n', 'gill-color_g',
       'gill-color_h', 'gill-color_k', 'gill-color_n', 'gill-color_p',
       'gill-color_r', 'gill-color_u', 'gill-color_w', 'gill-color_y',
       'stalk-shape_e', 'stalk-shape_t', 'stalk-root_b', 'stalk-root_c',
       'stalk-root_e', 'stalk-root_r', 'stalk-surface-above-ring_f',
       'stalk-surface-above-ring_k', 'stalk-surface-above-ring_s',
       'stalk-surface-above-ring_y', 'stalk-surface-below-ring_f',
       'stalk-surface-below-ring_k', 'stalk-surface-below-ring_s',
       'stalk-surface-below-ring_y', 'stalk-color-above-ring_b',
       'stalk-color-above-ring_c', 'stalk-color-above-ring_g',
       'stalk-color-above-ring_n', 'stalk-color-above-ring_p',
       'stalk-color-above-ring_w', 'stalk-color-above-ring_y',
       'stalk-color-below-ring_b', 'stalk-color-below-ring_c',
       'stalk-color-below-ring_g', 'stalk-color-below-ring_n',
       'stalk-color-below-ring_p', 'stalk-color-below-ring_w',
       'stalk-color-below-ring_y', 'veil-type_p', 'veil-color_w',
       'veil-color_y', 'ring-number_n', 'ring-number_o', 'ring-number_t',
       'ring-type_e', 'ring-type_l', 'ring-type_n', 'ring-type_p',
       'spore-print-color_h', 'spore-print-color_k', 'spore-print-color_n',
       'spore-print-color_r', 'spore-print-color_u', 'spore-print-color_w',
       'population_a', 'population_c', 'population_n', 'population_s',
       'population_v', 'population_y', 'habitat_d', 'habitat_g', 'habitat_l',
       'habitat_m', 'habitat_p', 'habitat_u'])


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


pca = PCA(n_components=90)


# In[ ]:


X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_


# In[ ]:


var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)


# In[ ]:


plt.pyplot.plot(var1)


# In[ ]:


pca1 = PCA(n_components=59)


# In[ ]:


X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_


# In[ ]:


var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)


# In[ ]:


var1


# In[ ]:


var3.mean()  #variance of the groups captured


# In[ ]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# In[ ]:


lr.fit(X_train,y_train)


# In[ ]:


print("Logistic Regressor Accuracy Score:", lr.score(X_test, y_test)*100)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


pred = lr.predict(X_test)


# In[ ]:


print(classification_report(pred,y_test))


# In[ ]:




