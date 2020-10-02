#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv', sep=',') # because our csv file has seperator ';' instead of ','
data.head()


# In[ ]:


print(data.keys())


# In[ ]:


data.shape


# In[ ]:


print(data.isna().sum())


# In[ ]:


data['quality'].unique()


# In[ ]:


data.quality.value_counts().sort_index()


# In[ ]:


sns.countplot(x='quality', data=data)
plt.show()


# In[ ]:


conditions = [ (data['quality'] >= 7),
              (data['quality'] <= 4)  ]

rating = ['superior', 'inferior']
data['rating'] = np.select(conditions, rating, default='fine')
data.rating.value_counts()


# In[ ]:


data.groupby('rating').mean()


# In[ ]:


correlations = data.corr()['quality'].drop('quality')
print(correlations)


# In[ ]:


plt.figure(figsize=(14,8))
sns.heatmap(data.corr(), annot=True, linewidths=0)
plt.show()


# ## Alcohol Percent vs Wine Quality

# In[ ]:


plt.figure()
Alcohol = sns.boxplot(x='quality', y='alcohol', data=data)

Alcohol.set(xlabel="Quality Dataset", ylabel="Alcohol Percent", 
            title="Percentage of Alcohol in different quality types")


# ## Sulphates vs Rating Dataset

# In[ ]:


plt.figure()
Sulphates = sns.boxplot(x='rating', y='sulphates', data=data)

Sulphates.set(xlabel='Rating Dataset', ylabel='Sulphates', title='Sulphates in different types of dataset ratings')


# In[ ]:


sns.lmplot(x='alcohol', y='residual sugar', col='rating', data=data)


# # Preparing Data for Modelling

# In[ ]:


dataset = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv', sep=',')
dataset.keys()


# In[ ]:


from sklearn.preprocessing import StandardScaler

X1 = dataset.drop(['quality'], axis=1)
X = StandardScaler().fit_transform(X1)

y = dataset['quality']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                    random_state=0)


# # Decision Tree

# In[ ]:


from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

model_one = DecisionTreeClassifier(random_state=1)
model_one.fit(X_train, y_train)


# In[ ]:


y_pred_one = model_one.predict(X_test)

print(classification_report(y_test, y_pred_one))


# # Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

model_two = RandomForestClassifier(random_state=1)
model_two.fit(X_train, y_train)

y_pred_two = model_two.predict(X_test)

print(classification_report(y_test, y_pred_two))


# # Feature Importance - via Random Forest 

# In[ ]:


plt.figure()
feat_importances = pd.Series(model_two.feature_importances_, index=X1.columns)
feat_importances.nlargest(25).plot(kind='barh', figsize=(5,5))

