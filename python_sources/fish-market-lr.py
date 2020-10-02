#!/usr/bin/env python
# coding: utf-8

# ## In this notebook I'll do some Linear Regression analysis on fish market data.. let's get started with importing our packages..

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ### Reading our dataset and getting initial insights..

# In[ ]:


df= pd.read_csv('/kaggle/input/fish-market/Fish.csv')
df.head()


# In[ ]:


df.tail()


# In[ ]:


df.describe()


# In[ ]:


df.shape


# ### We have 7 columns
# * Species: species name of fish
# * Weight: weight of fish in Gram g
# * Length1 : vertical length in cm
# * Length2 :diagonal length in cm
# * Length3 :cross length in cm
# * Height: height in cm
# * Width :diagonal width in cm
# 
# ### our target is to predict the weight of the fish based on the other features..
# 
# 
# 

# In[ ]:


df.Species.value_counts()


# In[ ]:


sns.catplot(x='Species' ,data=df,kind='count')


# ### Our data is imbalanced..

# In[ ]:


df.groupby('Species').median().sort_values('Weight',ascending=False)


# ### from the above table we notice that the features vary widely among the 8 fish species also the 'Roach' and 'Parkki' are kinda similar..

# ### Let's see a scatter plot to indicate th relation between features..

# In[ ]:


g=sns.pairplot(df,hue='Species')
plt.show()


# ### We can see that some features are highly correlated with each other so we'll drop some.
# ### I'll build model twice one with all the features and the other one with only Lenght1,weight and height

# In[ ]:


dfs=df[['Weight','Length1','Height','Width']]
std = StandardScaler()
dfs=pd.DataFrame(std.fit_transform(dfs),columns=dfs.columns)


# ### Since our dataset is really small I'll use cross validation..

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(dfs.drop('Weight',axis=1), dfs['Weight'], test_size=0.3, random_state=42)


# In[ ]:


lr=LinearRegression()
lr.fit(X_train,y_train)


# In[ ]:


print(lr.score(X_test,y_test) * 100)


# In[ ]:


yhat=lr.predict(X_test)
print(metrics.mean_squared_error(yhat,y_test))


# ### Trying another model, RandomForestRegressor

# In[ ]:


rf=RandomForestRegressor()
rf.fit(X_train,y_train)


# In[ ]:


print(rf.score(X_test,y_test) * 100)


# In[ ]:


yhatrf=rf.predict(X_test)
print(metrics.mean_squared_error(yhatrf,y_test))


# ### XGboost Regressor...

# In[ ]:


model_xgb = xgb.XGBRegressor(n_estimators=2000, max_depth=4, learning_rate=0.1, 
                             verbosity=1, silent=None, objective='reg:linear', booster='gbtree', 
                             n_jobs=-1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, 
                             subsample=0.8, colsample_bytree=0.8, colsample_bylevel=1, colsample_bynode=1, reg_alpha=0.2, reg_lambda=1.2, 
                             scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None, importance_type='gain')


# In[ ]:


model_xgb.fit(X_train, y_train)


# In[ ]:


yhatxgb= model_xgb.predict(X_test)


# In[ ]:


print(model_xgb.score(X_test,y_test) * 100)


# In[ ]:


print(metrics.mean_squared_error(yhatxgb,y_test))

