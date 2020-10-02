#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


# In[ ]:


https://www.kaggle.com/nadintamer/titanic-survival-predictions-beginner


# In[ ]:


df.shape


# In[ ]:


import pandas as pd
genderclassfare = pd.read_csv("../input/titanic-solution-for-beginners-guide/genderclassfare.csv")
test = pd.read_csv("../input/titanic-solution-for-beginners-guide/test.csv")
train = pd.read_csv("../input/titanic-solution-for-beginners-guide/train.csv")
df=train
df.head(2)


# In[ ]:


df.ndim


# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


df


# In[ ]:


from sklearn.preprocessing import LabelEncoder
lbe=LabelEncoder()
yeni_sex=lbe.fit_transform(df.Sex)
yeni_sex[:7]


# In[ ]:


df2=pd.DataFrame(yeni_sex,columns=["yeni_sex"])
df2[:2]


# In[ ]:


df=pd.concat([df,df2],axis=1)
df[0:3]


# In[ ]:


df.head(3)


# In[ ]:


df.size


# In[ ]:


df.tail(2)


# In[ ]:


df.drop("Cabin",axis=1,inplace=True)


# In[ ]:


df.isnull().values.any()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.Age.fillna(df.Age.mean(),inplace=True)


# In[ ]:


df.fillna(df.mean(),inplace=True)
df.isnull().sum()


# In[ ]:


df.head(2)


# In[ ]:


df.drop("Ticket", axis=1, inplace=True)
df.head(2)


# In[ ]:


df_one_hot=pd.get_dummies(df,columns=["Embarked"], prefix=["Embarked"])
df_one_hot.head(2)


# In[ ]:


df=df_one_hot.drop("Sex", axis=1, inplace=True)
df_one_hot


# In[ ]:


df=df_one_hot


# In[ ]:


df.head(2)


# In[ ]:


df_name=df.Name
df.head(2)


# In[ ]:


df.drop("Name", axis=1, inplace=True)


# In[ ]:


df[0:2]


# In[ ]:


import numpy as np
import pandas as pd 
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


predictors = df.drop(['Survived', 'PassengerId'], axis=1)
target = df["Survived"]
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.20, random_state = 42)


# In[ ]:


x_train.head(2)


# In[ ]:


get_ipython().system('pip install lightgbm')
get_ipython().system('conda install -c conda-forge lightgbm')
from lightgbm import LGBMClassifier
lgbm_model = LGBMClassifier().fit(X_train, y_train)
y_pred = lgbm_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))


# In[ ]:


lgbm = LGBMClassifier()


# In[ ]:


lgbm_params = {"learning_rate": [0.015, 0.014, 0.013,0.01],
              "n_estimators": [225,230,235,500],
              "max_depth":[18,18.5, 18.8,8]}
lgbm_cv_model = GridSearchCV(lgbm,lgbm_params, cv = 10, n_jobs = -1, verbose = 2).fit(X_train, y_train)
lgbm_cv_model.best_params_


# In[ ]:


lgbm_tuned = LGBMClassifier(learning_rate= 0.01, 
                            max_depth= 8, 
                            n_estimators= 500).fit(X_train, y_train)


# In[ ]:


y_pred = lgbm_tuned.predict(X_test)
accuracy_score(y_test, y_pred)


# In[ ]:


predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train["Survived"]
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.20, random_state = 0)


# In[ ]:


ids = test['PassengerId']
predictions = xgb_tuned.predict(test.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:




