#!/usr/bin/env python
# coding: utf-8

# # NHL Analytics Analysis
# 
# This is a brief analysis of NHL player salaries and some regression modeling using Light GBM as well as statistical analysis using statsmodels.
# 
# Since the data set is not a time based series we dont have to worry too much about serial correlation or heteroskedasticity.
# 
# The features of the data set are explained on the Kaggle page.
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))


# In[ ]:


train = pd.read_csv('../input/train.csv', encoding = "ISO-8859-1")
train.head(10)


# ## Clean Up

# In[ ]:


train.info()


# So there is 73 float features, 71 int, and 10 objects.

# In[ ]:


obj_cols = train.select_dtypes('object')
obj_cols.head()


# It looks like these are mostly straight forward and we will dig into them in a second but lets take care of the **Born** column.

# In[ ]:


obj_cols.drop('Born',axis=1,inplace=True)
train['Born'] = pd.to_datetime(train.Born)


# In[ ]:


train.Born.head()


# Now its into date time format and out of the object cols.

# In[ ]:


for c in obj_cols.columns:
    print('Obj Col: ', c, '   Number of Unqiue Values ->', len(obj_cols[c].value_counts()))


# In[ ]:


373+37+18+16+2+573+308+18+68


# If we dummied this out, we would get approximately an addition 1413 columns, minus the original 9.
# 
# We probably wouldnt need the first names though. It would be more interesting to keep the last names given some of hockey's family tradition.
# 
# There is something strange about the team category though since there is apparently 68 unique values...and currently only 31 hockey teams.

# ## A Little Exploration

# Lets take a look at the distribution of hockey players Countries relative to this data set.

# In[ ]:


fig, ax=plt.subplots(1,2,figsize=(18,10))
obj_cols['Cntry'].value_counts().sort_values().plot(kind='barh',ax=ax[0]) 
ax[0].set_title("Counts of Hockey Players by Country");
obj_cols['Cntry'].value_counts().plot(kind='pie', autopct='%.2f', shadow=True,ax=ax[1]);
ax[1].set_title("Distribution of Hockey Players by Country");


# Canada, USA, Sweden, Russia, Czekoslovakia, Finlad, etc.

# In[ ]:


fig, ax=plt.subplots(1,1,figsize=(12,8))
obj_cols['Team'].value_counts().plot(kind='bar',ax=ax);
plt.title('Counts of Team Values');


# So now we see what going on with the team values, there are actually some players who split teams so these are accounted for in this data set as well.

# ## Salary
# 
# Lets take a gander at whats going on with our target.

# In[ ]:


train.Salary.head(10)


# In[ ]:


fig, ax=plt.subplots(1,1,figsize=(12,8))
train.Salary.plot(kind='hist',ax=ax, bins=20);
plt.title("Distribution of Salaries");
plt.xlabel('Dollars');


# There are a lot of salalries less than a million.
# 
# This will very likely skew our data when we try to model it.
# 
# Lets take a look at salaries above a million.

# In[ ]:


sal_gtmil = train[train.Salary >= 1e7]


# In[ ]:


sal_gtmil.head(10)


# Right here we see some big names in this set. This set also only appears to be 7 items long.

# ## Model
# 
# My thoughts with this are to dummy out most of the objects variables and using light gbm.

# In[ ]:


import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# We had done some cleaning above but were going to create a function that handles it to work out the test data as well.
# 
# Additionally, we found out that the test and train splits arent evenly split so we need to go back and merge the data first.

# In[ ]:


def data_clean(x):
    ## Were going to change Born to date time
    x['Born'] = pd.to_datetime(x.Born, yearfirst=True)
    x['dowBorn'] = x.Born.dt.dayofweek
    x["doyBorn"] = x.Born.dt.dayofyear
    x["monBorn"] = x.Born.dt.month
    x['yrBorn'] = x.Born.dt.year
    ## Drop Pr/St due to NaNs from other countries and First Name
    x.drop(['Pr/St','First Name'], axis=1, inplace=True)
    ocols = ['City', 'Cntry', 'Nat', 'Last Name', 'Position', 'Team']
    for oc in ocols:
        temp = pd.get_dummies(x[oc])
        x = x.join(temp, rsuffix=str('_'+oc))
    x['Hand'] = pd.factorize(x.Hand)[0]
    x.drop(ocols, axis=1, inplace=True)
    x.drop(['Born'],axis=1,inplace=True)
    return x
    
    


# In[ ]:


try:
    del train, x0, xc, test
except:
    pass


# In[ ]:


train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")
train.head()


# In[ ]:


test = pd.read_csv('../input/test.csv', encoding="ISO-8859-1")
test.head()


# In[ ]:


full = train.merge(test, how='outer')
print(train.shape, test.shape, full.shape)


# In[ ]:


y = np.log(full.Salary.dropna())
full0 = full.drop(['Salary'],axis=1)


# In[ ]:


fig, ax=plt.subplots(1,1,figsize=(10,6))
y.plot(ax=ax);
plt.title("Ln Salary");


# In[ ]:


obj_cols.columns


# In[ ]:


full_c = data_clean(full0)


# In[ ]:


print(full0.shape, full_c.shape)


# In[ ]:


full_c.head()


# In[ ]:


ss = StandardScaler()


# In[ ]:


full_cs = ss.fit_transform(full_c)


# This cell below splits on the shape indices we have a few cells up.

# In[ ]:


train_c = full_cs[:612]
test_c = full_cs[612:]


# In[ ]:


print(train_c.shape, y.shape, test_c.shape)


# In[ ]:


type(y)


# ### LGB Model

# In[ ]:


folds = 3
lgbm_params = {
    "max_depth": -1,
    "num_leaves": 1000,
    "learning_rate": 0.01,
    "n_estimators": 1000,
    "objective":'regression',
    'min_data_in_leaf':64,
    'feature_fraction': 0.8,
    'colsample_bytree':0.8,
    "metric":['mae','mse'],
    "boosting_type": "gbdt",
    "n_jobs": -1,
    "reg_lambda": 0.9,
    "random_state": 123
}
preds = 0
for f in range(folds):
    xt, xv, yt, yv = train_test_split(train_c, y.values, test_size=0.2, random_state=((f+1)*123))
    
    xtd = lgb.Dataset(xt, label=yt)
    xvd = lgb.Dataset(xv, label=yv)
    mod = lgb.train(params=lgbm_params, train_set=xtd, 
                    num_boost_round=1000, valid_sets=xvd, valid_names=['valset'],
                    early_stopping_rounds=20, verbose_eval=20)
    
    preds += mod.predict(test_c)
    
preds = preds/folds
    
    


# ## Actual Test Data

# In[ ]:


acts = pd.read_csv('../input/test_salaries.csv', encoding="ISO-8859-1")
acts['preds'] = np.exp(preds)
acts.head()


# In[ ]:


import matplotlib
from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[ ]:


fig, ax=plt.subplots(1,1,figsize=(12,8))
acts.plot(ax=ax, style=['b-','r-']);
plt.title("Comparison of Preds and Actuals");
plt.ylabel('$');
ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.tight_layout()


# Based on the items above, it appears we missed a lot of the outliers which is something we would need to grab with perhaps a quantile regression.

# In[ ]:


mse = mean_squared_error(np.log(acts.Salary), np.log(acts.preds))
mae = mean_absolute_error(np.log(acts.Salary), np.log(acts.preds))


# In[ ]:


print("Ln Level Mean Squared Error :", mse)
print("Ln Level Mean Absolute Error :", mae)


# In[ ]:


fi_df = pd.DataFrame( 100*mod.feature_importance()/mod.feature_importance().max(), 
                      index=full_c.columns, #mod.feature_name(),
                      columns =['importance'])


# In[ ]:


fig, ax=plt.subplots(1,1,figsize=(12,8))
fi_df.sort_values(by='importance',ascending=True).iloc[-20:].plot(kind='barh', color='C0', ax=ax)
plt.title("Normalized Feature Importances");


# Top Features are:
# - Draft Year
# - Where the player was drafted over-all.
# - Year Born
# - Time On Ice / Games Played
# - Time On Ice (TOI/GP.l)
# - Draft Round
# - PLayers Avg Gane Score
# - Percentage of all opposing shot attempts blocked by this player
# - Expected goals (weighted shots) for this individual, which is shot attempts weighted by shot location
# - Day Of Year Born
# 
# 
# Time On Ice per GP doesnt seem so unusual as you would pay the ones who were better more and if theyre better, then they get paid more.
# 
# The Percentage of opposing shot attempts blocked is kinda of wild! Selke Trophy much?
# 

# In[ ]:


import statsmodels.api as sma


# For regressions I like to take the top couple effects from the Decision Tree and model it using statsmodels to get a better idea of the stastitical soundness of the model.

# In[ ]:


top10 = fi_df.sort_values(by='importance',ascending=True).iloc[-10:].index
top10

exog = pd.DataFrame(test_c, columns=full_c.columns)[list(top10)].fillna(0)


# In[ ]:


ols = sma.OLS(exog=exog, endog=acts.Salary)
ols_fit = ols.fit()
print(ols_fit.summary())


# According to the OLS, Draft Round and Block Shot % and Overall draft Position are not that statistically significant for this model.
# 
# The model has a good R squared.
# 

# In[ ]:


ols_preds = ols_fit.predict()


# In[ ]:


fig, ax=plt.subplots(1,1,figsize=(12,8))
acts.Salary.plot(ax=ax, color='C1');
ax.plot(ols_preds, color='C0');
plt.title("Comparison of StatsModels Preds and Actuals");
plt.ylabel('$');
plt.legend(['salary actual','ols preds']);
ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.tight_layout()


# In this instance, we were able to predict negative values which doesnt make sense and would have to be cut off at 0.

# In[ ]:




