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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import random
np.random.seed(0)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[ ]:


data=pd.read_csv('/kaggle/input/pokemon-challenge/pokemon.csv')


# In[ ]:


data.head()


# In[ ]:


del data['#']


# In[ ]:


data.head(10)


# In[ ]:


data.isnull().sum()


# In[ ]:


data=data.dropna(subset=['Name'])


# In[ ]:


data.isnull().sum()


# In[ ]:


data.loc[data['Type 2'].isnull()]


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(data['Type 2'])


# In[ ]:


data=data.fillna('#####')
type_02=['Poison', 'Dragon', 'Ground', 'Fairy', 'Grass',
       'Fighting', 'Psychic', 'Steel', 'Ice', 'Rock', 'Dark', 'Water',
       'Electric', 'Fire', 'Ghost', 'Bug', 'Normal']


# In[ ]:


data['Type 2']=data['Type 2'].apply(lambda x: x if x!='#####' else random.choice(type_02))


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(data['Type 2'])


# In[ ]:


data.head()


# In[ ]:


data['First_pokemon']=le.fit_transform(data['Name'])


# In[ ]:


data.head()


# In[ ]:


fight=pd.read_csv('/kaggle/input/pokemon-challenge/combats.csv')


# In[ ]:


print(fight.shape)
fight.head()


# In[ ]:


data_merged=data.merge(fight,on='First_pokemon',how='inner')


# In[ ]:


del data_merged['Name']


# In[ ]:


cat_cols=data_merged.select_dtypes(include='object')
cat_cols.head()


# In[ ]:


data_merged=pd.concat([data_merged.drop(cat_cols,axis=1),cat_cols.apply(le.fit_transform)],axis=1)


# In[ ]:


X=data_merged.drop('Winner',axis=1)
y=data_merged['Winner']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=5)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
rf=RandomForestRegressor()


# In[ ]:


rf.fit(x_train,y_train)


# In[ ]:


y_pred=rf.predict(x_test)


# In[ ]:


from sklearn.metrics import r2_score


# In[ ]:


r2_score(y_test,y_pred)


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV,GridSearchCV


# In[ ]:


param={'n_estimators':np.arange(1,10),'min_samples_split':np.arange(2,5),'min_samples_leaf':np.arange(1,10)}


# In[ ]:


search=GridSearchCV(estimator=rf,param_grid=param,return_train_score=True).fit(x_train,y_train)


# In[ ]:


search.best_params_


# In[ ]:


plt.figure(figsize=(10,7))
pd.DataFrame(search.cv_results_).set_index('params')['mean_test_score'].plot.line()
pd.DataFrame(search.cv_results_).set_index('params')['mean_train_score'].plot.line()
plt.xticks(rotation=90)
plt.show()


# In[ ]:


y_pred=search.predict(x_test)


# In[ ]:


r2_score(y_test,y_pred)


# In[ ]:


from xgboost import XGBRegressor


# In[ ]:


xgb=XGBRegressor()


# In[ ]:


xgb.fit(x_train,y_train)


# In[ ]:


y_pred=xgb.predict(x_test)


# In[ ]:


r2_score(y_test,y_pred)

