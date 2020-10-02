#!/usr/bin/env python
# coding: utf-8

# # Libs

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from scipy import stats
from scipy.stats import norm 
import warnings 

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit, RepeatedStratifiedKFold
from sklearn import metrics
import gc
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from bayes_opt import BayesianOptimization

warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.metrics.scorer import make_scorer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

import xgboost as xgb
from lightgbm import LGBMRegressor
import gc


# # Load datasets

# In[ ]:


df_train = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip', parse_dates=['Date'])
df_test = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip', parse_dates=['Date'])

df_stores = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv')
df_features = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv.zip', parse_dates=['Date'])


# ## Looking at data

# In[ ]:


df_train.head()


# In[ ]:


df_stores.head()


# In[ ]:


df_features.head()


# ## Joining the datasets

# In[ ]:


#train
df_train_join = df_train.merge(df_stores, on='Store', how='left')
df_train_join = df_train_join.merge(df_features, on=['Store', 'Date', 'IsHoliday'], how='left')

#test
df_test_join = df_test.merge(df_stores, on='Store', how='left') 
df_test_join = df_test_join.merge(df_features, on=['Store', 'Date', 'IsHoliday'], how='left')


# ## Check

# In[ ]:


df_train_join.head()


# In[ ]:


df_test_join.head()


# # EDA

# ## First look on our dataset

# In[ ]:


df_train_join.shape


# In[ ]:


df_train_join.dtypes


# Let's transform some of out features into categorical values

# In[ ]:


df_train_join.describe()


# Negative values on weekly sales needs to be corrected or neglected. Let's see if this erros occurs on many rows.

# In[ ]:


len(df_train_join[df_train_join['Weekly_Sales'] < 0])/len(df_train_join)


# This erros occurs on less than 1% of the dataset. Phew.

# Checking missing values:

# In[ ]:


df_train_join.isnull().sum(axis = 0)


# In[ ]:


df_test_join.isnull().sum(axis = 0)


# Train set:
# - We only have null values on MarkDown columns. Those columns represents the Type of markdown and what quantity was available during that week.
# 
# Test set:
# - Null values on CPI and Unemployment rate

# In[ ]:


set(df_test_join[~df_test_join.Unemployment.notnull()]['Date'])


# In[ ]:


set(df_test_join[~df_test_join.CPI.notnull()]['Date'])


# All missing values on test set are from 2013.

# In[ ]:


df_train_join[df_train_join.isnull().any(axis=1)]['Date']


# Missing values on years 2010, 2011 and 2012 on out train set.

# In[ ]:


df_train_join.skew()


# Markdown columns are skewed features.

# ## Data Exploration

# Let's see our sales distribution per day:

# In[ ]:


sns.kdeplot(df_train_join['Weekly_Sales'])


# In[ ]:


print("Skewness: {} \nKurtosis: {}".format(df_train_join['Weekly_Sales'].skew().round(), df_train_join['Weekly_Sales'].kurt().round()))


# Hightly skewed, big tail (high kurtosis). Lets create more datetime columns:

# In[ ]:


#train
df_train_join['Year'] = df_train_join['Date'].dt.year
df_train_join['Month'] = df_train_join['Date'].dt.month
df_train_join['Week'] = df_train_join['Date'].dt.week
df_train_join['Day'] = df_train_join['Date'].dt.day

#test
df_test_join['Year'] = df_test_join['Date'].dt.year
df_test_join['Month'] = df_test_join['Date'].dt.month
df_test_join['Week'] = df_test_join['Date'].dt.week
df_test_join['Day'] = df_test_join['Date'].dt.day


# In[ ]:


from pylab import rcParams
rcParams['figure.figsize'] = 8, 8


# For now, let's plot just one of them to see if there's an impact of sales.

# In[ ]:


sns.boxplot(x="Month", y="Weekly_Sales", data=df_train_join, showfliers=False)


# We can see some seasonality here.

# ## What about hollidays?

# In[ ]:


sns.boxplot(x="IsHoliday", y="Weekly_Sales", data=df_train_join, showfliers=False)


# In[ ]:


df_train_join[df_train_join['IsHoliday']==True]['Week'].unique()


# - We have some indications of more sales on holidays, but not much.
# - We see hollidays on weeks 6 (superbowl), 36 (labor day), 47 (thanksgiving) and 52 (christmas)
# - It is missing Easter!

# ## Sales per year

# In[ ]:


sns.lineplot(x="Week", y="Weekly_Sales", data=df_train_join, hue='Year')


# - Peak of yearly sales on the last weeks/holliday related
# - Sales on year before seems like a strong predictor (may become a feature)
# - Easter looks like an important holliday on 1st semester, we should include it. Our first weak model could be simply the weekly sales iin the year before +- some threshold, quick win :)

# ## Sales per store

# In[ ]:


sns.boxplot(y="Weekly_Sales", x="Store", data=df_train_join, showfliers=False)


# Each store has its own sales pattern.

# ## Sales per department type

# In[ ]:


sns.boxplot(x="Dept", y="Weekly_Sales", data=df_train_join, showfliers=False)


# Big difference between department types. 

# ## Sales per store type

# In[ ]:


sns.boxplot(y="Type", x="Weekly_Sales", data=df_train_join, showfliers=False)


# Visually, we get a hint that there are some stores selling more than others, and store type is algo important: Store type C tends to sell less than store type B or A.

# ## What if we combine store type and department?

# In[ ]:


plt.figure(figsize=(10, 50))
sns.boxplot(y='Dept', x='Weekly_Sales', data=df_train_join, showfliers=False, hue="Type",orient="h") 


# We can wee clear distinction of weekly sales just by oversing those two variables combined.

# ## Does store size matter?

# In[ ]:


sns.boxplot(x='Size', y='Weekly_Sales', data=df_train_join, showfliers=False)


# Yes it does. The smaller the store, the smaller the sales.

# ## Store type distribution

# In[ ]:


df_train_join[['Store','Type']].groupby(['Type']).nunique()


# The distribution of store types is not equal. We have more store types that tends to have more weekly sales.

# ## Sales per unenployment rate

# In[ ]:


sns.scatterplot(x="Weekly_Sales", y="Unemployment", data=df_train_join)


# We can't see much here. One could argue that higher unemployment rates would affect weekly sales, but we cannot see that clearly on the plot. Most of weekly sales comes from average unemployment rate locations, but that only means that most of the locations have average unenployment rates.

# # Correlations

# We already looked into some features, but let's see how they correlate to each other:

# In[ ]:


corr = df_train_join.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr, annot=True, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# We can have some new takeaways and confirm what we already saw on the previous plots:
# - Temperature no correlation with weekly sales, but strong correlation with variable Month (as expected, of course)
# - The bigger the store, more weekly sales
# - Fuel price does not seem to have correlation with weekly sales
# - CPI and unemployment rate (with missing values on test set) have low correlation rate with weekly sales

# # Data preprocessing

# Let's use all out previous conclusions to move forward.

# ## Handling missing values

# In[ ]:


#MarkDown1 to Markdown5: null values = 0
df_train_join[['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']] = df_train_join[['MarkDown1', 'MarkDown2', 'MarkDown3', 
                                                                                                  'MarkDown4', 'MarkDown5']].fillna(value=0)

df_test_join[['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']] = df_test_join[['MarkDown1', 'MarkDown2', 'MarkDown3', 
                                                                                                  'MarkDown4', 'MarkDown5']].fillna(value=0)
#MarkDown1 to Markdown5: if less than 0 = 0
df_train_join['MarkDown1'] = df_train_join['MarkDown1'].apply(lambda x: 0 if x < 0 else x)
df_train_join['MarkDown2'] = df_train_join['MarkDown2'].apply(lambda x: 0 if x < 0 else x)
df_train_join['MarkDown3'] = df_train_join['MarkDown3'].apply(lambda x: 0 if x < 0 else x)
df_train_join['MarkDown4'] = df_train_join['MarkDown4'].apply(lambda x: 0 if x < 0 else x)
df_train_join['MarkDown5'] = df_train_join['MarkDown5'].apply(lambda x: 0 if x < 0 else x)

df_test_join['MarkDown1'] = df_test_join['MarkDown1'].apply(lambda x: 0 if x < 0 else x)
df_test_join['MarkDown2'] = df_test_join['MarkDown2'].apply(lambda x: 0 if x < 0 else x)
df_test_join['MarkDown3'] = df_test_join['MarkDown3'].apply(lambda x: 0 if x < 0 else x)
df_test_join['MarkDown4'] = df_test_join['MarkDown4'].apply(lambda x: 0 if x < 0 else x)
df_test_join['MarkDown5'] = df_test_join['MarkDown5'].apply(lambda x: 0 if x < 0 else x)

# Negative weekly sales on train data
df_train_join['Weekly_Sales'] = df_train_join['Weekly_Sales'].apply(lambda x: 0 if x < 0 else x)

# CPI and Unemployment rate - drop (low correlation + missing values on test set)
# We can - always - change our mind later if model performance is poor)
df_train_join.drop(['CPI', 'Unemployment'], axis=1, inplace=True)
df_test_join.drop(['CPI', 'Unemployment'], axis=1, inplace=True)

# Fuel price and Temperature - Drop (low value on correlation plot)
df_train_join.drop(['Fuel_Price', 'Temperature'], axis=1, inplace=True)
df_test_join.drop(['Fuel_Price', 'Temperature'], axis=1, inplace=True)


# In[ ]:


# Drop date - we transformed it into year/month/day/week variables
df_train_join.drop(['Date'], axis=1, inplace=True)
df_test_join.drop(['Date'], axis=1, inplace=True)


# ## Feature engineering

# In[ ]:


#log for skewed variables
skewed = ['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']
df_train_join[skewed] = df_train_join[skewed].apply(lambda x: np.log(x + 1))
df_test_join[skewed] = df_test_join[skewed].apply(lambda x: np.log(x + 1))

#adding easter holiday - we saw on the EDA it was important!
df_train_join.loc[(df_train_join.Year==2010) & (df_train_join.Week==13), 'IsHoliday'] = True
df_train_join.loc[(df_train_join.Year==2011) & (df_train_join.Week==16), 'IsHoliday'] = True
df_train_join.loc[(df_train_join.Year==2012) & (df_train_join.Week==14), 'IsHoliday'] = True

df_test_join.loc[(df_test_join.Year==2013) & (df_test_join.Week==13), 'IsHoliday'] = True

#create dummy variables - store type and holliday
df_train_join = pd.get_dummies(df_train_join, columns=['Type'])
df_test_join = pd.get_dummies(df_test_join, columns=['Type'])

#trasform IsHoliday
df_train_join['IsHoliday'] = df_train_join['IsHoliday'].apply(lambda x: 1 if x==True else 0)
df_test_join['IsHoliday'] = df_test_join['IsHoliday'].apply(lambda x: 1 if x==True else 0)


# In[ ]:


#Also apply log to weekly sales
df_train_join['Weekly_Sales'] = df_train_join['Weekly_Sales'].apply(lambda x: np.log(x + 1))


# # Data split (train-test-validation)

# In[ ]:


train_X, val_X, train_y, val_y = train_test_split(df_train_join.drop('Weekly_Sales', axis = 1), 
                                                  df_train_join['Weekly_Sales'], 
                                                  test_size = 0.2, 
                                                  random_state = 42)


# # Chosen model: Ensemble with tree based models

# Why?
# - Robust to outliers
# - Proven performance on several competitions
# - Easy interpretability for tree based models (for non-technical stakeholders)
# - Can handle both categorical and numerical data :)
# 
# Watch out for:
# - Overfitting!!!!

# ## Chosen metric: Weighted Mean Absolute Error (WMAE)

# It was the chosen metric for this competition.
# It punishes mistakes based on some weight criteria, specially on hollidays.

# In[ ]:


def wmae(pred_y, test_y, weights):
    return 1/sum(weights) * sum(weights * abs(test_y - pred_y))

def calculate_weights(holidays):
    return holidays.apply(lambda x: 1 if x==0 else 5)


# # Train and predict base model

# In[ ]:


def train_predict(model, train_X, train_y, test_X, test_y, verbose=0): 
    
    results = {}  
    
    model = model.fit(train_X, train_y)
    predictions = model.predict(test_X)
            
    # WMAE on Test Set
    results['WMAE'] = wmae(np.exp(test_y), 
                           np.exp(predictions), 
                           calculate_weights(test_X['IsHoliday']))
    

    importances = model.feature_importances_
    std = np.std([model.feature_importances_ for model in model.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]
    #Print importances
    for f in range(train_X.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, train_X.columns[f], importances[indices[f]]))

    # Success
    print("Model Name:", model.__class__.__name__)
    print("WMAE:", round(results['WMAE'],2))
    
    # Return the model & predictions
    return (model, predictions, importances)


# In[ ]:


model_rf = RandomForestRegressor(random_state=42, verbose=1)


# In[ ]:


m, pred_y, feature_importances = train_predict(model_rf, train_X, train_y, val_X, val_y, verbose=3)


# Here we can see which variables affects our model the most :)
# First feature seems highly correlated to the outcome, which is often **undesirable**.

# # Tuning using Grid Search CV

# In[ ]:


params = { 
    'n_estimators': [50, 80],
    'max_features': [None, 'auto'],
    'bootstrap': [True, False],
    'max_depth':[None],
    'random_state': [42], 
    'verbose': [1]
}


# If you have more time/computing power, you can test lots other parameters/values.

# In[ ]:


CV = GridSearchCV(estimator=model_rf, param_grid=params, cv=3, verbose=1)
CV.fit(train_X, train_y)


# In[ ]:


print('Best parameters: {}'.format(CV.best_params_))


# In[ ]:


final_model = model_rf.set_params(**CV.best_params_)


# In[ ]:


m_final, pred_y_val_final, feature_importances_final = train_predict(final_model, train_X, train_y, val_X, val_y, verbose=3)


# # Plotting results

# In[ ]:


y_test_pred = m_final.predict(df_test_join)


# In[ ]:


df_test_join['Weekly_Sales'] = y_test_pred


# In[ ]:


sns.lineplot(x="Week", y="Weekly_Sales", data=df_test_join)


# # Important remarks

# Tunned model feature importances didn't change its magnitude.
# If WMAE is good enough, the model that goes into production uses this parameters!
# <br>
# <br>
# What dictates if the model is good enough overall? **Business needs** (output speed + computing cost + some WMAE threshold)
# <br>
# <br>
# What if the error is still too big?
# - Retrain w/o higly correlated feature
# - Use more parameters on grid search or use Bayesian Optimization (+ computing cost)
# - New features e.g countdown for important hollidays (xmas, thanksgiving...)
# - Boosting models such as lightgbm, catboost, xboost (it decreases the explainability and needs more computing power, but a  lot more powerful)
