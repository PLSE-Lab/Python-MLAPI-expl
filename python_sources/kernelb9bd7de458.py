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
for dirname, _, filenames in os.walk('/kaggle/input/TTiDS20'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# In[ ]:


cars = pd.read_csv("/kaggle/input/TTiDS20/train.csv")
cars.info()


# In[ ]:


cars["type"].fillna("No Type", inplace = True)
cars["gearbox"].fillna("No Type", inplace = True)
cars['gearbox'].astype('category').value_counts()
cars["fuel"].fillna("No Type", inplace = True)
cars["engine_capacity"].fillna(cars["engine_capacity"].median(), inplace = True)
cars["damage"].fillna(cars["damage"].median(), inplace = True)
cars["insurance_price"].fillna(cars["insurance_price"].median(), inplace = True)






cars.head()


# In[ ]:


cars_numeric=cars.select_dtypes(include=['float64','int64'])
cars_numeric.head()


# In[ ]:


plt.figure(figsize=(20, 10))
sns.pairplot(cars_numeric)


# In[ ]:


for i, col in enumerate(cars_numeric.columns):
    plt.figure(i)
    sns.scatterplot(x=cars_numeric[col],y=cars_numeric['price'])


# In[ ]:


carnames = cars['brand'].apply(lambda x: x.split(" ")[0])
carnames[:15]


# In[ ]:


X = cars.drop(columns=['price'])
y = cars['price']


# In[ ]:


cars_categorical = X.select_dtypes(include=['object'])
cars_categorical.head(2)


# In[ ]:


cars_dummies = pd.get_dummies(cars_categorical, drop_first=True)
cars_dummies.head()


# In[ ]:


X = X.drop(columns=cars_categorical)
X.head(2)


# In[ ]:


X = pd.concat([X,cars_dummies],axis=1)
X.head(2)


# In[ ]:


from sklearn.preprocessing import scale

# storing column names in cols, since column names are (annoyingly) lost after 
# scaling (the df is converted to a numpy array)
cols=X.columns
X=pd.DataFrame(scale(X))
X.columns=cols
X.columns


# In[ ]:


X.describe()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.7,
                                                    test_size = 0.3, random_state=100)


# In[ ]:


from sklearn import linear_model
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train,y_train)

y_pred_test=lm.predict(X_test)
y_pred_train=lm.predict(X_train)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
from sklearn.metrics import r2_score

print('{}'.format(mean_absolute_percentage_error(y_true=y_train, y_pred=y_pred_train)))
print('{}'.format(mean_absolute_percentage_error(y_true=y_test, y_pred=y_test)))

#Standard error/RMSE
error_train=y_pred_train-y_train
error_test=y_pred_test-y_test

print('RMSE on train data: {}'.format(((error_train**2).mean())**0.5))
print('RMSE on test data: {}'.format(((error_test**2).mean())**0.5))


# In[ ]:


cars_test = pd.read_csv("/kaggle/input/TTiDS20/test_no_target.csv")
cars_test.info()

cars_test["type"].fillna("No Type", inplace = True)
cars_test["gearbox"].fillna("No Type", inplace = True)
cars_test['gearbox'].astype('category').value_counts()
cars_test["fuel"].fillna("No Type", inplace = True)
cars_test["engine_capacity"].fillna(cars_test["engine_capacity"].median(), inplace = True)
cars_test["damage"].fillna(cars_test["damage"].median(), inplace = True)
cars_test["insurance_price"].fillna(cars_test["insurance_price"].median(), inplace = True)

X_test = cars_test


# In[ ]:


cars_categorical_test = X_test.select_dtypes(include=['object'])
cars_categorical_test.head(2)


# In[ ]:


cars_dummies_test = pd.get_dummies(cars_categorical_test, drop_first=True)
cars_dummies_test.head()
cars_dummies_test.info()


# In[ ]:


X_test = X_test.drop(columns=cars_categorical_test)
X_test.head(2)


# In[ ]:


X_test = pd.concat([X_test,cars_dummies_test],axis=1)
X_test.head(2)


# In[ ]:


from sklearn.preprocessing import scale

# storing column names in cols, since column names are (annoyingly) lost after 
# scaling (the df is converted to a numpy array)
cols=X_test.columns
X_test=pd.DataFrame(scale(X_test))
X_test.columns=cols
print(((X.columns).tolist()) - ((X_test.columns).tolist())


# In[ ]:


y_pred_test=lm.predict(X_test)
print (y_pred_test)

