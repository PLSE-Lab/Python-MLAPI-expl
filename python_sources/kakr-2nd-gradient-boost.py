#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# > ### train & test data

# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[ ]:


print(len(df_train.columns))
print(len(df_test.columns))


# > ### target variable: price

# In[ ]:


df_train['price'].describe()


# In[ ]:


sns.distplot(df_train['price']);


# #### skewness and kurtosis of price variable

# In[ ]:


print("Skewness: %f" % df_train['price'].skew())
print("Kurtosis: %f" % df_train['price'].kurt())


# ## Price vs variables

# In[ ]:


plt.figure(figsize = (8, 5))
sns.jointplot(df_train.sqft_living, df_train.price, 
              alpha = 0.5)
plt.xlabel('sqft_living')
plt.ylabel('price')
plt.show()


# In[ ]:


plt.figure(figsize = (8, 5))
sns.jointplot(df_train.sqft_living15, df_train.price, 
              alpha = 0.5)
plt.xlabel('sqft_living15')
plt.ylabel('price')
plt.show()


# In[ ]:


plt.figure(figsize = (8, 5))
sns.jointplot(df_train.sqft_lot, df_train.price, 
              alpha = 0.5)
plt.xlabel('sqft_lot')
plt.ylabel('price')
plt.show()


# In[ ]:


plt.figure(figsize = (8, 5))
sns.jointplot(df_train.sqft_lot15, df_train.price, 
              alpha = 0.5)
plt.xlabel('sqft_lot15')
plt.ylabel('price')
plt.show()


# In[ ]:


plt.figure(figsize = (8, 5))
sns.jointplot(df_train.bedrooms, df_train.price, 
              alpha = 0.5)
plt.xlabel('bedrooms')
plt.ylabel('price')
plt.show()


# In[ ]:


plt.figure(figsize = (8, 5))
sns.jointplot(df_train.bathrooms, df_train.price, 
              alpha = 0.5)
plt.xlabel('bathrooms')
plt.ylabel('price')
plt.show()


# In[ ]:


plt.figure(figsize = (8, 5))
sns.jointplot(df_train.view, df_train.price, 
              alpha = 0.5)
plt.xlabel('view')
plt.ylabel('price')
plt.show()


# In[ ]:


plt.figure(figsize = (8, 5))
sns.jointplot(df_train.grade, df_train.price, 
              alpha = 0.5)
plt.xlabel('grade')
plt.ylabel('price')
plt.show()


# In[ ]:


plt.figure(figsize = (8, 5))
sns.jointplot(df_train.condition, df_train.price, 
              alpha = 0.5)
plt.xlabel('condition')
plt.ylabel('price')
plt.show()


# In[ ]:


plt.figure(figsize = (8, 5))
sns.jointplot(df_train.floors, df_train.price, 
              alpha = 0.5)
plt.xlabel('floors')
plt.ylabel('price')
plt.show()


# In[ ]:


plt.figure(figsize = (8, 5))
sns.jointplot(df_train.lat, df_train.price, 
              alpha = 0.5)
plt.xlabel('lat')
plt.ylabel('price')
plt.show()


# In[ ]:


plt.figure(figsize = (8, 5))
sns.jointplot(df_train.long, df_train.price, 
              alpha = 0.5)
plt.xlabel('long')
plt.ylabel('price')
plt.show()


# ## Correlation between price & variables

# In[ ]:


corrmat = df_train.corr()
colormap = plt.cm.RdBu
plt.figure(figsize=(16,14))
plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(corrmat, fmt='.2f',linewidths=0.1, vmax=0.9, square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 10})


# In[ ]:


corrmat.sort_values(["price"], ascending = False, inplace = True)
print(corrmat.price)


# In[ ]:


#scatterplot
sns.set()
cols = ['price', 'sqft_living', 'sqft_above', 'view', 'grade', 'bathrooms', 'sqft_living15']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();


# In[ ]:


#transformed histogram and normal probability plot
sns.distplot(df_train['price'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['price'], plot=plt)


# In[ ]:


#applying log transformation
df_train['price'] = np.log(df_train['price'])


# In[ ]:


#transformed histogram and normal probability plot
sns.distplot(df_train['price'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['price'], plot=plt)


# In[ ]:


df_train.columns


# In[ ]:


df_test.columns


# In[ ]:


df_train['price_per_ft'] = df_train['price'] / df_train['sqft_lot']
price_with_zipcode = df_train.groupby(['zipcode'])['price_per_ft'].agg({'mean', 'std'}).reset_index()

df_train = pd.merge(df_train, price_with_zipcode, how='left', on='zipcode')
df_test = pd.merge(df_test, price_with_zipcode, how='left', on='zipcode')
del df_train['price_per_ft']


# In[ ]:


df_train.columns


# In[ ]:


df_test.columns


# In[ ]:


df_train['date']


# In[ ]:


for df in [df_train, df_test]:
    df['date(new)'] = df['date'].apply(lambda x: int(x[4:8])+800 if x[:4] == '2015' else int(x[4:8])-400)
    del df['date']
    df['floor_area_ratio'] = df['sqft_living'] / df['sqft_lot']
    df['rooms'] = df['bedrooms'] + df['bathrooms']
    del df['sqft_lot15']

    
# Log Scaling
log_features = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15',
                'mean', 'std',
               ]
for df in [df_train, df_test]:
    for feature in log_features:
        df[feature] = np.log1p(df[feature])


# In[ ]:


df_train['date(new)']


# In[ ]:


df_train.columns


# In[ ]:


X_train = df_train.drop(['id', 'price'], axis=1)
y_train = df_train['price']
X_test = df_test.drop(['id'], axis=1)


# In[ ]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, AdaBoostRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb


# > ### Light GBM

# > ### GBM

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor


# In[ ]:


GBoost_model = GradientBoostingRegressor(n_estimators=8000, learning_rate=0.05,
                                   max_depth=5, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =4)
GBoost_model.fit(X_train, y_train)


# In[ ]:


GBoost_prediction = np.expm1(GBoost_model.predict(X_test))
print(GBoost_prediction.shape)
GBoost_prediction[:20]


# In[ ]:


sub = pd.DataFrame({'id':df_test['id'],'price':GBoost_prediction})
sub.to_csv('submission_GBoost.csv',index=False)

