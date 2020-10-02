#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.style as style
# style.available
import seaborn as sns
style.use('fivethirtyeight')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Any results you write to the current directory are saved as output.


# In[ ]:


style.available


# In[ ]:


import os
print(os.listdir("../input"))
df = pd.read_csv('../input/diabetes.csv')


# In[ ]:


get_ipython().run_cell_magic('HTML', '', '<style type="text/css">\ntable.dataframe td, table.dataframe th {\n    border: 2px  black solid !important;\n    color: black !important;\n}\n</style>')


# In[ ]:





# In[ ]:


df.columns


# In[ ]:


df.describe()


# In[ ]:


df.head()


# In[ ]:


print("Todtal number of dataset in diabetes data:{} ".format(df.shape))


# In[ ]:


print(df.groupby('Outcome').size())


# In[ ]:


sns.countplot(df['Outcome'],label="Count")


# In[ ]:


sns.relplot(x = "Age", y= "Glucose", sort = False, kind="line", data=df)


# In[ ]:


sns.distplot(df["Glucose"], bins=20, kde=False, rug=True);


# In[ ]:


sns.distplot(df["Glucose"], hist=False, rug=True)


# In[ ]:


sns.regplot(x= df["Glucose"], y = df["Outcome"], data=df )


# In[ ]:


plt.figure(figsize=[10,10])
sns.lmplot(x= "Glucose", y = "Outcome", data=df )


# In[ ]:


sns.lmplot(x= "Age", y = "BMI", data=df )


# In[ ]:


sns.regplot(x= "Age", y = "BMI", data=df )


# In[ ]:


f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, linewidths=0.5, ax=ax)


# In[ ]:


f, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(x="Age", y="BMI",
            hue="Outcome", palette=["m", "g"],
            data=df)
sns.despine(offset=10, trim=True)


# In[ ]:


f, ax = plt.subplots(figsize=(10, 8))
sns.barplot(x="Age", y="BMI",
            hue="Outcome", palette=["m", "g"],
            data=df)
sns.despine(offset=10, trim=True)


# In[ ]:


f, ax = plt.subplots(figsize=(10, 8))
sns.swarmplot(x="Age", y="Insulin",
            hue="Outcome", palette=["m", "g"],
            data=df)
sns.despine(offset=10, trim=True)


# In[ ]:


f, ax = plt.subplots(figsize=(16, 10))
sns.boxplot(x="Age", y="Insulin",
            hue="Outcome", palette=["m", "g"],
            data=df)
sns.despine(offset=10, trim=True)


# In[ ]:


color_list = ['red' if i == 'Abnormal' else 'green' for i in df.loc[:, 'Outcome']]
pd.plotting.scatter_matrix(df.loc[:, df.columns != 'Outcome'],
                           c = color_list,
                           figsize=[15,15],
                           diagonal='hist',
                           alpha = 1.0,
                           s = 200,
                           marker = "*",
                           edgecolor = 'black')
plt.show()


# In[ ]:


cases = pd.DataFrame(df.groupby('Age')['Outcome'].sum()).head(10)
cases


# In[ ]:


df1 = df.loc[:, df.columns != 'Outcome']
df1


# 1. ## Compare Machine Learning Algorithms
# * Logistic Regression.
# *  Linear Discriminant Analysis.
# *  k-Nearest Neighbors.
# *  Classi cation and Regression Trees.
# *  Naive Bayes.
# *  Support Vector Machines.

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


df_data = df.values
df_data


# In[ ]:


X = df_data[:, 0:8]
Y = df_data[:, 8]
# prepare models
models = []
names = []
results = []
scoring = 'accuracy'
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=test_size,random_state=7)
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print(name, cv_results.mean(), (cv_results.std()))


# In[ ]:


fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




