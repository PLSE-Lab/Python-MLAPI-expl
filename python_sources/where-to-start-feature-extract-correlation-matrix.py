#!/usr/bin/env python
# coding: utf-8

# **If you are as new as me, why don't you start your House Pricing Model with me.  **  
# **Created by Raymond Wang **
# 
# **If you are as new as me Series **     
# Titanic https://www.kaggle.com/yw6916/if-you-are-as-new-as-me-why-don-t-you-start-here1?scriptVersionId=15917018    
# House Pricing Advanced https://www.kaggle.com/yw6916/house-pricing-advance-if-you-are-as-new-as-me-3  (Part 2 of this model, please go and check it out if you want something to improve yours' performance)
# 
# In this Kernal, I will lead you through the journey of making a vanilla model with a straight and simple approach. I hope you will enjoy this very intuitive method.

# **What you will learn from here: **   
# 1. General Kaggle Data Science workflow
# 2. Numerical and Non-numerical data analysis and handling
# 3. Feature Extraction and engineering
# 4. Heat map, correlation matrix for systematic analysis.
# 5. Machine learning techniques and how to validate models.
# 6. Missing data handling.
# 

# Standard Import  
# We always start a notebook with import. These are libraries that people usually use for data science.

# In[ ]:


#standard import
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# Read the dataset, and use head( ) to preview, this is a good way to understand data generally.

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df    = pd.read_csv("../input/test.csv")

# preview the data
train_df.head()


# In[ ]:


#train_df.info()
#print("----------------------------")
#test_df.info()


# In almost any situation, id is an irrelevant feature to prediction. Therefore, we drop it.

# In[ ]:


#drop unnecessary data
train_df=train_df.drop(['Id'],axis=1)
train_df.head()


# Until this point, we have almost the right dataset to sail. Then, in the following section, different methods are taken to further analyze the data.

# **1.Target Analysis**

# For the first step, it is always a good idea to start with analyzing target, in this case, the SalePrice.

# In[ ]:


train_df['SalePrice'].describe()


# In[ ]:


sns.distplot(train_df['SalePrice'])


# The target here will not assist a lot to increase accuracy in this model. However, for advanced settings, skewness can be used to adjust outliers. Please refer to https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard.

# **2. Features Extraction**

# In this section, I mannually pick 8 features which I think is critical to this analysis. Then, I will examine all of them to see if they are actually usefull and relevant for prediction. It is a very good exercise to build data scientist intuition.

# In this sections, the 8 features extracted are:       
# 1.LotArea   
# 2.GrLivArea    
# 3.OverallQual   
# 4.Utilities   
# 5.Neighborhood     
# 6.CentralAir    
# 7.GarageCars&GarageArea    
# 8.YearBuilt    

# 2.1 LotArea

# In[ ]:


# The intuition here is that the larger the area is, the higher the price should be.
# Therefore, we are examing this proportionality in this regards.
data = pd.concat([train_df['SalePrice'], train_df['LotArea']], axis=1)
data.plot.scatter(x='LotArea', y='SalePrice', ylim=(0, 800000))

#Well, not really see a proportionality in the chart, probably drop it.


# 2.2.GrLivArea 

# In[ ]:


# Similar intuition as LotArea
data = pd.concat([train_df['SalePrice'], train_df['GrLivArea']], axis=1)
data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0, 800000))
#Yeah, in this case, the linear proportionality is quite obvious. Keep it.
#However, we can further clean the data by removing outliers.
train_df = train_df.drop(train_df[(train_df['GrLivArea']>4000) & (train_df['SalePrice']<300000)].index)


# 2.3.OverallQual 

# In[ ]:


data = pd.concat([train_df['SalePrice'], train_df['OverallQual']], axis=1)
#data.plot.scatter(x='OverallQual', y='SalePrice', ylim=(0, 800000))
#better version of visualization
fig = sns.boxplot(x='OverallQual', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
#It is a quite good index. Definitely keep it.


# 2.4.Utilities  

# In[ ]:


#dropped. All the same
pass


# 2.5.Neighborhood 

# In[ ]:


data = pd.concat([train_df['SalePrice'], train_df['Neighborhood']], axis=1)
f, ax = plt.subplots(figsize=(26, 12))
fig = sns.boxplot(x='Neighborhood', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
# Different Neighborhoods have different range of price. Keep it.


# Neighborhood is proven to be less relevent. However, for learning purposes, I will demonstrate how to deal with non-numerical data.

# In[ ]:


#Neighborhood is proven to be less effective, ignore.

# Using Dummies to extract this feature
#neighborhood_dummies_train  = pd.get_dummies(train_df['Neighborhood'])
#neighborhood_dummies_test  = pd.get_dummies(test_df['Neighborhood'])
#train_df = train_df.join(neighborhood_dummies_train)
#test_df    = test_df.join(neighborhood_dummies_test)

#train_df.drop(['Neighborhood'], axis=1,inplace=True)
#test_df.drop(['Neighborhood'], axis=1,inplace=True)

#train_df.head()


# 2.6.CentralAir 

# In[ ]:


data = pd.concat([train_df['SalePrice'], train_df['CentralAir']], axis=1)
f, ax = plt.subplots()
fig = sns.boxplot(x='CentralAir', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
#With CentralAir, the sale price is higher. Keep it for now


# In[ ]:


#Another way to handle non-numerical data.


# In[ ]:


#convert Y,N into 1,0
train_df['CentralAir'].replace(to_replace=['N', 'Y'], value=[0, 1])
train_df.head()


# 2.7.GarageCars&GarageArea

# In[ ]:


# This can be seen as a factor to represent the number of cars owning.
# Typically, more cars, the house tends to be more expensive
# We select GarageCars due to personal preference only
data = pd.concat([train_df['SalePrice'], train_df['GarageCars']], axis=1)
data.plot.scatter(x='GarageCars', y='SalePrice', ylim=(0, 800000))
data = pd.concat([train_df['SalePrice'], train_df['GarageArea']], axis=1)
data.plot.scatter(x='GarageArea', y='SalePrice', ylim=(0, 800000))
# Fairly representative, keeping it


# 2.8.YearBuilt  

# In[ ]:


# This may be a tricky one to see the correlation, since time series is involvd.
data = pd.concat([train_df['SalePrice'], train_df['YearBuilt']], axis=1)
data.plot.scatter(x='YearBuilt', y='SalePrice', ylim=(0, 800000))
#for better visualization
f, ax = plt.subplots(figsize=(26, 12))
fig = sns.boxplot(x='YearBuilt', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
#index is okay, but too many outliers, think about it later.


# **3. Overall Analysis**     
# 

# After process 2, we gain an general idea about how indeces can affect our prediction. However, the above features are selected neither scientific nor systematic. Then, in this section, I will introduce heatmap, a systematic approach in finding right features.

# In here, I introduce a new and general method to handle non-numerical data with sklearn.

# In[ ]:


#This sk method can process non-value datat= like Neighborhood
from sklearn import preprocessing
f_names = ['CentralAir', 'Neighborhood']
for x in f_names:
    label = preprocessing.LabelEncoder()
    train_df[x] = label.fit_transform(train_df[x])


# To drwa heatmap for correlation. The fainter the colour, the higher the correlation. Now, by focusing the bottom row named SalePrice, we can find the correlation between our targets and features.

# In[ ]:


corrmat =train_df.corr()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)


# From the heat map, we extract missing potential useful index:     
# 1.FullBath.    
# 2.TotRmsAbvGrd.   
# 3.TotalBsmtSF.   
# 4.1stFlrSF.   
# Also, Neighborhood and CentralAir seem to be less relevant. So drop them.

# Then, use correlation matrix to examin the top 10 features to eliminate similar ones.

# In[ ]:


k  = 10 
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True,                  square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# From this matrix, 1stFlrSF and TotalBsmtSF are similar. We take 1stFlrSF.
# GarageCars and GarageArea are similar. We takeGarageCars.
# TotRmsAbvGrd and GrLivArea are similar. We take GrLivArea.

# Therefore, the model now only consists the following index, which we believed to be most correlated from the map: 
# 1. OveralQual
# 2. GrLivArea
# 3. GarageArea
# 4. 1stFlrSF
# 5. FullBath
# 6. YearBuilt
# 
# 

# **4. ML to generate Models**

# For regression problems, we usually take staking approch (a very vanila one here, just to introduce the idea). 

# ML import with different models.

# In[ ]:


from sklearn import preprocessing
from sklearn import linear_model, svm, gaussian_process
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


# Process the data with our 6 top chosen features.  
# Then using normalization to do feature scaling.
# Also, we split data into training and validation set.

# In[ ]:


cols = ['OverallQual','GrLivArea', 'GarageArea','1stFlrSF', 'FullBath', 'YearBuilt']
x = train_df[cols].values
y = train_df['SalePrice'].values
#Normalization
x_scaled = preprocessing.StandardScaler().fit_transform(x)
y_scaled = preprocessing.StandardScaler().fit_transform(y.reshape(-1,1))
#Train and validation
X_train,X_vali, y_train, y_vali = train_test_split(x_scaled, y_scaled, test_size=0.3, random_state=42)


# We inspect 3 example ML models: RandomForest, KNN and LGBoost (for the idea of dropping poorly performed one).  
# A very naive method of validation is used to test models.
# 

# In[ ]:


cols = ['OverallQual','GrLivArea', 'GarageArea','1stFlrSF', 'FullBath', 'YearBuilt']
X_train = train_df[cols].values
y_train = train_df['SalePrice'].values

clf_1 = RandomForestRegressor(n_estimators=400)
clf_1.fit(X_train, y_train)
y_pred = clf_1.predict(X_vali)
print(np.sum(abs(y_pred - y_vali))/len(y_pred))

clf_2 = KNeighborsRegressor(n_neighbors=7)
clf_2.fit(X_train, y_train)
y_pred = clf_2.predict(X_vali)
print(np.sum(abs(y_pred - y_vali))/len(y_pred))

clf_3 = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

clf_3.fit(X_train, y_train)
y_pred = clf_3.predict(X_vali)
print(np.sum(abs(y_pred - y_vali))/len(y_pred))


# We can see, the off-set of LGBoost is highest. Therefore, we discard LGBoost.

# **5.Missing Data Handling**

# After training our model, we have to think about our input. Is the dataset complete?

# In[ ]:


cols = ['OverallQual','GrLivArea', 'GarageArea','1stFlrSF', 'FullBath', 'YearBuilt']
test_df[cols].isnull().sum()


# No, there is one missing data in GarageArea we have to handle by filling with mean value.

# In[ ]:


#Handling GarageArea Missing data
test_df['GarageArea'].describe()


# In[ ]:


test_df['GarageArea']=test_df['GarageArea'].fillna(472.768861)
test_df['GarageArea'].isnull().sum()


# **6. Submission**

# Then, we are ready to go for submitting this vanila model!
# I will in the following days to summerize techniques of optimising this model, and I will upload a link here. If you like my idea, please stay with me. Thanks!!!

# In[ ]:


cols = ['OverallQual','GrLivArea', 'GarageArea','1stFlrSF', 'FullBath', 'YearBuilt']
test_x = pd.concat( [test_df[cols]] ,axis=1)

x = test_x.values

y_pred_1 = clf_1.predict(x)
y_pred_2 =clf_2.predict(x)
y_pred_3 =clf_3.predict(x)

y_pred=y_pred_1*0.5+y_pred_2*0.5


# In[ ]:


prediction = pd.DataFrame(y_pred, columns=['SalePrice'])
print(prediction)
result = pd.concat([test_df['Id'], prediction], axis=1)
result.to_csv('./Predictions.csv', index=False)

