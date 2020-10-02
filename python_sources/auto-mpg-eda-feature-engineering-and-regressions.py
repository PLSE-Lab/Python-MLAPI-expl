#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# When I started doing this analysis my main goal was getting experience. I'm still learning and trying to improve my skills, so there might be some areas can be improved. My main objectives on this project are:
# * **Explorating and visualising the data with pandas and seaborn packages**
# * **Building and tuning couple regression models to get some stable results with sklearn and xgboost packages**
# 
# ## Data Set Information:
# 
# This dataset is a slightly modified version of the dataset provided in the StatLib library. In line with the use by Ross Quinlan (1993) in predicting the attribute "mpg", 8 of the original instances were removed because they had unknown values for the "mpg" attribute. The original dataset is available in the file "auto-mpg.data-original".
# 
# "The data concerns city-cycle fuel consumption in miles per gallon, to be predicted in terms of 3 multivalued discrete and 5 continuous attributes." (Quinlan, 1993)
# 
# 
# ## Attribute Information:
# 
# 1. mpg: continuous
# 2. cylinders: multi-valued discrete
# 3. displacement: continuous
# 4. horsepower: continuous
# 5. weight: continuous
# 6. acceleration: continuous
# 7. model year: multi-valued discrete
# 8. origin: multi-valued discrete
# 9. car name: string (unique for each instance)
# 

# * **So let's begin with importing neccesary libraries:**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#

from sklearn.preprocessing import LabelEncoder, StandardScaler


# In[ ]:


train_df = pd.read_csv('/kaggle/input/autompg-dataset/auto-mpg.csv')


# ### Exploratory Data Analysis
# 
# **Exploratory data analysis would be a great start for us. We need to get some insights before building our models.**

# In[ ]:


print(f'Training Shape: {train_df.shape}')


# In[ ]:


print(train_df.head())
print(train_df.sample(5))


# In[ ]:


display(train_df.info())


# **When we take a look at basic statistics we can already see that horsepower column has datatype of object but it should be a float type. In order to convert this column first we need to replace missing value "?" in this data to "NaN".**

# In[ ]:


train_df['horsepower'] = train_df['horsepower'].replace('?', np.NaN).astype('float64')


# **Before we replace missing values in horsepower let's check correlations between hp and other variables:**

# In[ ]:


train_df_corr = train_df.corr().abs().unstack().sort_values(kind='quicksort', ascending=False).reset_index()
train_df_corr.rename(columns={"level_0": "Feature A", 
                             "level_1": "Feature B", 0: 'Correlation Coefficient'}, inplace=True)
train_df_corr[train_df_corr['Feature A'] == 'horsepower'].style.background_gradient(cmap='summer_r')


# **We are going to fill missing values with grouping more related columns.**

# In[ ]:


train_df['horsepower'] = train_df.groupby(['displacement'], sort=False)['horsepower'].apply(lambda x: x.fillna(x.mean()))
train_df['horsepower'] = train_df.groupby(['cylinders'], sort=False)['horsepower'].apply(lambda x: x.fillna(x.mean()))


# **Seems better. Distribution is skewed but we'll deal with it later...**

# In[ ]:


sns.set()
plt.subplots(figsize=(10, 6))
sns.distplot(train_df['horsepower'],bins=25)
plt.show()


# In[ ]:


print(f'There are some missing values: {train_df.isna().any().any()}')


# **When we look at figure below we notice there is some degree of relation between some of the features by just looking it. Interesting...**

# In[ ]:


g = sns.PairGrid(train_df.drop('car name',axis=1), hue='origin')
g = g.map_diag(plt.hist, alpha=0.4)
g = g.map_upper(sns.scatterplot)
g = g.map_lower(sns.regplot)


# * **Before we move on, I just wanted to take look at effect of origin on the vehicles mpg since our database is mostly US(Around 62%):**

# In[ ]:


org=train_df.copy()
org['origin']=train_df.origin.map({1: 'US', 2: 'Asian',3:'European'})
org['origin'].value_counts(normalize=True)


# **From this boxplot we can easily see that US manufactured cars are least efficent in terms of mpg and they are almost below global average. It might mean nothing though, we need to move on:**

# In[ ]:


sns.set()
fig, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='origin', y="mpg", data=org)
plt.axhline(org.mpg.mean(),color='r',linestyle='dashed',linewidth=2)
plt.show()


# **Our heatmap confirms there is some kind of linear relation between these features. Let's see if we can catch this trend with our regression models. We'll be there soon...**

# plt.figure(figsize=(12,6))
# sns.heatmap(train_df.corr(),annot=True, linewidths=0.2,cmap='coolwarm', center=0)
# plt.xticks(rotation=45)
# plt.yticks(rotation=45)
# plt.show()

# # Data Processing & Feature Engineering

# **Before we move on to build our regression models we might take something out of car names feature instead of just dropping them.**
# 
# * **It looks like our database has the car brands in first words of our car name column, can we extract this out for searching relations between them?**

# In[ ]:


pd.crosstab(train_df['car name'],train_df['origin'])


# In[ ]:


train_df['Brand'] = train_df['car name'].str.extract('([A-Za-z]+)\s', expand=False)


# **Our dataset has some typos and duplicate names for some brands, we must replace them before we proceed.**

# In[ ]:


train_df['Brand']= train_df['Brand'].replace(np.NaN, 'subaru')
train_df['Brand']= train_df['Brand'].replace('chevroelt', 'chevrolet')
train_df['Brand']= train_df['Brand'].replace('vw', 'volkswagen')
train_df['Brand']= train_df['Brand'].replace('toyouta', 'toyota')
train_df['Brand']= train_df['Brand'].replace('vokswagen', 'volkswagen')
train_df['Brand']= train_df['Brand'].replace('maxda', 'mazda')
train_df['Brand']= train_df['Brand'].replace('mazada', 'mazda')
train_df['Brand']= train_df['Brand'].replace('chevy', 'chevrolet')


# In[ ]:


train_df['Brand'].value_counts(normalize=True)


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 8))
sns.countplot(train_df['Brand'])
plt.xticks(rotation=60)
plt.show()


# **It looks much better! It's time to transform these brands to model applicable. We going to categorize them by using sklearn's Label Encoder:** 

# In[ ]:


le = LabelEncoder()
train_df['Brand'] = le.fit_transform(train_df['Brand'])
train_df.drop('car name', axis=1, inplace=True)
train_df.sample(5)


# **Before building our models last thing I want to check is if our data distribution skewed.**

# In[ ]:


features=train_df.columns.tolist()
for feature in features:
    print(f'{feature} Skewness: {train_df[feature].skew():.2f}, Kurtosis: {train_df[feature].kurtosis():.2f}')


# **Can we decrease our skewness with some log scaling? Well little bit better than before.**

# In[ ]:


skew_cols=['cylinders','displacement','horsepower','weight']
train_df[skew_cols]=np.log1p(train_df[skew_cols])
for feature in features:
    print(f'{feature} skewness: {train_df[feature].skew():.2f}, Kurtosis: {train_df[feature].kurtosis():.2f}')


# # Modelling

# In[ ]:


from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, BaggingRegressor
import xgboost as xgb
import lightgbm as lgb


#  **Let's start with cross validation, then we take a closer look on these models**

# In[ ]:


n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_df.drop('mpg', axis=1))
    rmse= np.sqrt(np.abs(cross_val_score(model, train_df.drop('mpg', axis=1).values, train_df['mpg'], scoring="neg_mean_squared_error", cv = kf, n_jobs=-1)))
    return(rmse)

def rtw_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_df.drop('mpg', axis=1))
    rtw= cross_val_score(model, train_df.drop('mpg', axis=1).values, train_df['mpg'], scoring="r2", cv = kf, n_jobs=-1)
    return(rtw)


# **Without tuning hyper-paramaters our best option looks like GradientBoostingRegressor, but while in terms of being very close to GBR, XGBRegressor does little bit better on r2 score.**

# In[ ]:


mods = [LinearRegression(),Ridge(),GradientBoostingRegressor(),
      RandomForestRegressor(),BaggingRegressor(),
      xgb.XGBRegressor(), lgb.LGBMRegressor()]

model_df = pd.DataFrame({
    'Model': [type(i).__name__ for i in mods],
    'RMSE': [np.mean(rmsle_cv(i)) for i in mods],    
    'Rmse Std': [np.std(rmsle_cv(i)) for i in mods],
    'R2': [np.mean(rtw_cv(i)) for i in mods],
    'R2 Std': [np.std(rmsle_cv(i)) for i in mods]})
display(model_df.sort_values(by='RMSE', ascending=True).reset_index(drop=True).style.background_gradient(cmap='summer_r'))


# **I'm also going to do ta train test split to take a closer look on models and check feature importances on different models.**

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train_df.drop('mpg', axis=1),train_df['mpg'], test_size= 0.33, random_state=42)


# In[ ]:


linreg=LinearRegression()
linreg.fit(X_train,y_train)
y_pred=linreg.predict(X_test)

rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)
print(r2_score(y_test, y_pred))


# In[ ]:


ridge = Ridge()
ridge.fit(X_train,y_train)
y_pred=ridge.predict(X_test)

rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)
print(r2_score(y_test, y_pred))


# In[ ]:


bag_regressor = BaggingRegressor()
bag_regressor.fit(X_train,y_train)
y_predict = bag_regressor.predict(X_test)
rmse_bgr = np.sqrt(mean_squared_error(y_test,y_predict))

rmse=np.sqrt(mean_squared_error(y_test,y_predict))
print(rmse)
print(r2_score(y_test, y_predict))


# In[ ]:


gb_regressor = GradientBoostingRegressor()
gb_regressor.fit(X_train,y_train)
y_predict = gb_regressor.predict(X_test)
rmse_bgr = np.sqrt(mean_squared_error(y_test,y_predict))

rmse=np.sqrt(mean_squared_error(y_test,y_predict))
print(rmse)
print(r2_score(y_test, y_predict))


feature_imp = pd.DataFrame(sorted(zip(gb_regressor.feature_importances_,train_df.drop('mpg', axis=1).columns)), columns=['Value','Feature'])

plt.figure(figsize=(12, 6))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('Features')
plt.tight_layout()
plt.show()


# In[ ]:


rf_regressor = RandomForestRegressor()
rf_regressor.fit(X_train,y_train)
y_predict = rf_regressor.predict(X_test)
rmse_bgr = np.sqrt(mean_squared_error(y_test,y_predict))

rmse=np.sqrt(mean_squared_error(y_test,y_predict))
print(rmse)
print(r2_score(y_test, y_predict))


feature_imp = pd.DataFrame(sorted(zip(rf_regressor.feature_importances_,train_df.drop('mpg', axis=1).columns)), columns=['Value','Feature'])

plt.figure(figsize=(12, 6))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('Features')
plt.tight_layout()
plt.show()


# In[ ]:


xg_reg=xgb.XGBRegressor(booster='gbtree', objective='reg:squarederror')

xg_reg.fit(X_train,y_train)
xg_y_pred=xg_reg.predict(X_test)
xg_rmse=np.sqrt(mean_squared_error(y_test,xg_y_pred))
print(xg_rmse)
print(r2_score(y_test,xg_y_pred))

feature_imp = pd.DataFrame(sorted(zip(xg_reg.feature_importances_,train_df.drop('mpg', axis=1).columns)), columns=['Value','Feature'])

plt.figure(figsize=(12, 6))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('Features')
plt.tight_layout()
plt.show()


# In[ ]:


lgb_reg = lgb.LGBMRegressor()

lgb_reg.fit(X_train,y_train)
y_predict=lgb_reg.predict(X_test)
rmse=np.sqrt(mean_squared_error(y_test,y_predict))
print(rmse)
print(r2_score(y_test,y_predict))

feature_imp = pd.DataFrame(sorted(zip(lgb_reg.feature_importances_,train_df.drop('mpg', axis=1).columns)), columns=['Value','Feature'])

plt.figure(figsize=(12, 6))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('Features')
plt.tight_layout()
plt.show()


# **I could say our model did ok with the small data we have!**
# 
# **Thank you for checking out my work! I'm in my early days of this journey, there is lot to learn. Please let me know if I did some mistakes, I'm still trying to improve myself!**
# 
