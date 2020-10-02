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


import seaborn as sns
from matplotlib import pyplot as plt
from pandas.plotting import andrews_curves, parallel_coordinates, radviz
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df0 = pd.read_csv("../input/iris/Iris.csv")


# In[ ]:


df = df0.copy()


# In[ ]:


df.head()


# In[ ]:


df.drop(labels = "Id" , axis = 1, inplace = True)


# In[ ]:


m = np.random.uniform(1,7, size = (16,4))
df1 = pd.DataFrame(m, columns = ["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"])
n = pd.DataFrame({'Species' : ["Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-setosa","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-versicolor","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica","Iris-virginica"]}, columns = ["Species"])


# In[ ]:


df1 = pd.concat([df1,n], axis = 1)
df = pd.concat([df1,df], ignore_index = True)


# In[ ]:


df.groupby("Species").describe().T


# In[ ]:


df.groupby("Species").quantile([.25,0.75])


# ## Data Visualization

# In[ ]:


for i in df.columns[0:-1]:
    sns.boxplot(x='Species', y = i , data = df)
    plt.show()


# In[ ]:


sns.pairplot(df, hue = 'Species', size = 3)


# In[ ]:


andrews_curves(df, "Species")


# In[ ]:


parallel_coordinates(df, "Species")


# In[ ]:


radviz(df, "Species")


# ## 3 Sigma

# In[ ]:


for column in df.columns[0:-1]:
    for spec in df["Species"].unique():
        selected_spec = df[df["Species"] == spec ]
        selected_column = selected_spec[column]
        
        std = selected_column.std()
        avg = selected_column.mean()
        
        three_sigma_plus = avg + (std * 3)
        three_sigma_minus = avg - (std * 3)
        
        outliers = selected_column[((selected_spec[column] > three_sigma_plus) | (selected_spec[column] < three_sigma_minus))].index
        df.drop(index = outliers, inplace = True)
        
        print(outliers)


# ## Quantile

# In[ ]:


for column in df.columns[0:-1]:
    for spec in df["Species"].unique():
        selected_spec = df[df['Species'] == spec ]
        selected_column = selected_spec[column]
        
        q1 = selected_column.quantile(.25)
        q3 = selected_column.quantile(.75)
        
        iqr = q3 - q1
        mini = q1 - (1.5 * iqr)
        maxi = q3 + (1.5 * iqr)
        
        max_idxs = df[(df['Species'] == spec ) & (df[column] > maxi)].index
        min_idxs = df[(df['Species'] == spec ) & (df[column] < mini)].index
        
        df.drop(index = max_idxs, inplace = True)
        df.drop(index = min_idxs, inplace = True)

        
        print(max_idxs)
        print(min_idxs)


# ## Label Encoding

# In[ ]:


le = LabelEncoder()
df["Species"] = le.fit_transform(df["Species"])


# In[ ]:


y = df["Species"]
X = df.drop(["Species"], axis = 1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)


# In[ ]:


y_train.value_counts()


# In[ ]:


y_test.value_counts()


# ## XGBoost

# In[ ]:


xgb_cls = xgb.XGBClassifier(objective = "multiclass:softmax", num_class = 3)
xgb_cls.fit(X_train, y_train)


# In[ ]:


preds = xgb_cls.predict(X_test)


# In[ ]:


accuracy_score(y_test, preds)


# ## Model Tuning

# In[ ]:


xgb_params = {
    'n_estimators' : [100, 500, 1000, 2000],
    'subsample' : [0.6 , 0.8 , 1.0],
    'max_depth' : [3, 4, 5, 6]}


# In[ ]:


xgb_cv_model = GridSearchCV(xgb_cls, xgb_params, cv = 10, n_jobs = -1, verbose = 2)
xgb_cv_model.fit(X_train, y_train)


# In[ ]:


xgb_cv_model.best_params_


# In[ ]:


xgb_cls = xgb.XGBClassifier(max_depth = 3,
                       n_estimators = 100,
                       subsample = 0.8)


# In[ ]:


xgb_tuned = xgb_cls.fit(X_train,y_train)


# In[ ]:


y_pred = xgb_tuned.predict(X_test)
accuracy_score(y_test,y_pred)


# In[ ]:


confusion_matrix(y_test,y_pred)


# In[ ]:





# In[ ]:




