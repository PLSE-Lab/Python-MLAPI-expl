#!/usr/bin/env python
# coding: utf-8

# # Online News Popularity Data Set 
# This dataset is provided by [UCI ML Dataset](http://archive.ics.uci.edu/ml/datasets/Online+News+Popularity).  It has got details about 39K articles published in Mashables and number of shares each article has got. The idea is to build a model to predict the number of shares. Before doing so, we will also try to perform certan feature selection and also perform EDA
# 
# First of all, let's get the data from UCI dataset. 
# 

# In[ ]:


get_ipython().system('wget http://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip -O OnlineNewsPopularity.zip')
get_ipython().system('yes | unzip OnlineNewsPopularity.zip')


# The details of dataset is provided below

# In[ ]:


get_ipython().system('cat OnlineNewsPopularity/OnlineNewsPopularity.names')


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

##  SKlearn libs for regressions
from sklearn import metrics, linear_model
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

##Importing Libraries for Neural Nets
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation

import xgboost


# # Data Load and Understanding

# Load the data and print shape of the data.

# In[ ]:


df = pd.read_csv('./OnlineNewsPopularity/OnlineNewsPopularity.csv')
df.shape


# We can see there are 39.6K data records with 61 columns. 1 target variable
# 
# Let us see how the data look like

# In[ ]:


df.head()


# Let us try to analyse null values in the input

# In[ ]:


# gives some infos on columns types and number of null values
tab_info=pd.DataFrame(df.dtypes).T.rename(index={0:'column type'})
tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'null values (nb)'}))
tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100)
                         .T.rename(index={0:'null values (%)'}))
tab_info


# We can observe that, there are no null values in any of the columns, which is great news.. Let us now try to remove unwanted and non-value adding columns

# In[ ]:


cols_to_remove = ['url']
df = df.drop(['url'], axis=1)
df.head()


# ## Data Cleaning, Outlier treatment
# 
# ### No. of Shares
# Let us analyse how the sharing data is spread

# In[ ]:


plt.subplots(3,1,figsize=(20,16))
plt.subplot(3,1,1)
sns.distplot(df[' shares'], hist=True, kde=False)
plt.subplot(3,1,2)
sns.violinplot(df[' shares'])
plt.subplot(3,1,3)
sns.scatterplot(data=df, x=' timedelta', y=' shares')


# We can see that the data is very skewed. We can also observe that there are very low no. of articles with very large number of shares. The 3rd scatter plot confirms the same. 
# 
# So, next let us try to find and remove the **outliers**

# In[ ]:


Q1 = df[' shares'].quantile(0.25)
Q3 = df[' shares'].quantile(0.75)
IQR = Q3 - Q1
LTV= Q1 - (1.5 * IQR)
UTV= Q3 + (1.5 * IQR)
df = df.drop(df[df[' shares'] > UTV].index)
df.shape


# After removing the outliers, let us try to plot the same graphs

# In[ ]:


plt.subplots(3,1,figsize=(20,16))
plt.subplot(3,1,1)
sns.distplot(df[' shares'], hist=True, kde=False)
plt.subplot(3,1,2)
sns.violinplot(df[' shares'])
plt.subplot(3,1,3)
sns.scatterplot(data=df, x=' timedelta', y=' shares')


# Now the data looks really good for next steps.. 

# # Feature selection
# We have seen that there are around 60 features. In this section, we analyse statistically and find which are all the important features to consider

# ### Correlation Analysis

# In[ ]:


df_corr = abs(df.corr())
df_corr = df_corr[' shares']
df_corr = pd.DataFrame(df_corr.values, df.columns).reset_index()
# print(df_corr[0:20],df_corr[21:40], df_corr[41:]  )
df_corr.columns = ['Feature', 'Corr']
df_corr = df_corr[df_corr['Corr'] > 0.06]
df_corr = df_corr.sort_values(by='Corr', ascending=False)[1:]
df_corr = df_corr.head(20)
df_corr


# Above list shows the list of features which has more than 6% correlation (P Value) with respect to no. of shares

# ### Feature selection using *Univariate Selection*
# 

# Let us now split the data into X and Y, where Xs are all the features and Y is the *no. of shares*

# In[ ]:


y = df[' shares']
X = df.drop([' shares'], axis=1)

bestfeatures = SelectKBest(score_func=f_classif, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
featureScores = featureScores.sort_values(by='Score', ascending=False).head(20)
featureScores


# Now we have found top 20 features based on *correlation* and *ANOVA*. Let us compare and see which are common in both

# In[ ]:


idx1 = pd.Index(df_corr['Feature'])
idx2 = pd.Index(featureScores['Specs'])
features_list = idx1.intersection(idx2)
features_list


# Above are common features in both Corr and ANOVA. We can use these features for building the model
# 
# Next step, let us analyse if any of there features are highly correlated to each other using a heatmap. We can keep one of highly correlated features.

# ### Features shortlisting 

# In[ ]:


X1 = X[features_list]
plt.subplots(1,1,figsize=(18,8))
sns.heatmap(X1.corr(),annot=True,cmap="RdYlGn")


# Green patches other than diagonal are the once to check and consider one of them. Following pairs are correlated features. The highlighted ones are features whose scores are higher in either **Correlation** and/or **ANOVA**
# 
# - **kw_avg_avg** - kw_max_avg
# - **kw_avg_avg** - kw_min_avg
# - **is_weekend** - weekday_is_saturday
# - **data_channel_is_world** - LDA_02
# - **data_channel_is_entertainment** - LDA_01
# - **data_channel_is_tech** - LDA_04 
# 
# Following features should be remvoed from above list: kw_max_avg, kw_min_avg, weekday_is_saturday, LDA_02, LDA_01, LDA_04
# 
# 

# In[ ]:


features_to_remove = [' kw_max_avg',' kw_min_avg',' weekday_is_saturday',' LDA_02',' LDA_01',' LDA_04']
lst = list(features_list.values)
selected_features = [e for e in lst if e not in features_to_remove]

## Let us take top 6
# selected_features = selected_features[:6]

selected_features 


# By using two features selection techniques we found 20 top features and then we selected common features among them and then removed correlated features. We get the list of features as shown above. They are in the order of importance 

# # Exploratory Data Analysis
# Let us now perform some deep exploratory data analysis

# ### Shares vs kw_avg_avg
# This is feature which is most impacting, as per statistics. Let us see how it looks

# In[ ]:


plt.subplots(1,1,figsize=(10,8))

plt.subplot(1,1,1)

plt.title('Scatter shares vs kw_avg_avg')
sns.scatterplot(data=df, x=' kw_avg_avg', y=' shares', ci=None)


# We can see that articles published on weekend is getting shared more than weekday. The same is explained below

# In[ ]:


plt.subplots(2,1,figsize=(10, 8))

plt.subplot(2,1,1)
plt.title('No. of articles shares during weekend')
sns.barplot(data=df, x=' is_weekend', y=' shares', ci=False)

plt.subplot(2,1,2)
plt.title('No. of articles during weekend')
sns.distplot(df[' is_weekend'], hist=True, kde=False)


# ### No. of shares vs weekday of the article
# Let us get a new column with weekday to see how the shares are varying with that column

# In[ ]:


index = 100
df['weekday'] = df[' weekday_is_monday'] * 2 ** 1 +       df[' weekday_is_tuesday'] * 2 ** 2 +       df[' weekday_is_wednesday'] * 2 ** 3 +       df[' weekday_is_thursday'] * 2 ** 4 +       df[' weekday_is_friday'] * 2 ** 5 +       df[' weekday_is_saturday'] * 2 ** 6 +       df[' weekday_is_sunday'] * 2 ** 7 
    
# np.log2(val)
df.head()


# In[ ]:


plt.subplots(1,1,figsize=(10, 6))

plt.subplot(1,1,1)
plt.title('No. of articles shares by no. of keywords')
sns.barplot(data=df, x=' num_keywords', y=' shares', ci=False)


# We can see that no. of shares increase with key words in the meta data

# Next up we will see how *no. of shares* and *rate_negative_words* featrues are correlated

# In[ ]:


fig, ax = plt.subplots(1,1,figsize=(10,8))
ax = fig.add_subplot(1, 1, 1)
# ax.set_yscale('log')
plt.title('Feature: rate_negative_words')
sns.scatterplot(data=df, x=' rate_negative_words', y=' shares', ci=None)    


# The above scatterplot is very sparse, but we can observe that there is a correlation

# In[ ]:


def token_group(x):
    if x < 100:
        return 1
    elif x >=100 and x<500:
        return 2
    elif x >=500 and x<1000:
        return 3
    elif x >=1000 and x<2000:
        return 4
    else: 
        return 5
    
df['token_group'] =  df[' n_tokens_content'].apply(token_group)


# In[ ]:


plt.subplots(2,1,figsize=(10,15))

plt.subplot(2,1,1)
plt.title('No. of articles in token group')
sns.distplot(df.token_group, hist=True, kde=False)
plt.subplot(2,1,2)
plt.title('Avg shares in each group')
sns.barplot(data=df, x='token_group', y=' shares', ci=None)


# We have most of the articles in group 2, which is with token length in the range of 100 to 500. As the articles size is increasing, we see avg no. shares has also increased

# # Build Model
# 
# ### Baseline Model
# We have ```selected_features``` as part of feature selection section. Let us try to build naive (Baseline) model based on simple linear model
# 
# First of all, let us split the train and test sets for the model

# In[ ]:


X = X[selected_features]


# In[ ]:


# print(X[selected_features].shape)
# print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[ ]:


model = linear_model.LinearRegression()
model.fit(X_train, y_train)


# We use MSE and R2 as Metrics

# In[ ]:


results = model.predict(X_test)
score1 = metrics.mean_squared_error(y_test,results )
score2 = metrics.r2_score(y_test,results )
print('MSE: ',score1, '  R2 Score: ', score2)


# As expected the R2 score is pretty bad. 
# 

# In[ ]:


poly = PolynomialFeatures(degree = 2)
regr = linear_model.LinearRegression()
X_ = poly.fit_transform(X_train)
regr.fit(X_, y_train)
X_ = poly.fit_transform(X_test)
results = regr.predict(X_)
score1 = metrics.mean_squared_error(y_test,results )
score2 = metrics.r2_score(y_test,results )
print('MSE: ',score1, '  R2 Score: ', score2)


# Polynominal algorithm could not finish as 9 features, its too much to calculate the multiple possible poly degrees
# However, in one of the previous versions, we have run and following is the output 
# 
# ```
# Model with Polynominal Degree 1 MSE:  1138058.1341509074   R2 Score:  0.08366289295516138
# Model with Polynominal Degree 2 MSE:  1128850.3833187486   R2 Score:  0.09107675302675189
# Model with Polynominal Degree 3 MSE:  1131198.7487819889   R2 Score:  0.08918590549419136
# Model with Polynominal Degree 4 MSE:  1133594.5157797942   R2 Score:  0.08725689138318526
# Model with Polynominal Degree 5 MSE:  1140495.5739477822   R2 Score:  0.08170032490609536
# Model with Polynominal Degree 6 MSE:  2199982.8449245617   R2 Score:  -0.7713734080645245
# ```
# 
# 

# ### Random Forest Regressor
# We will use Gridsearch with cross validation to get best parameter for RandomForest Algo

# In[ ]:



# # model2 = RandomForestRegressor(n_estimators=5, max_depth=1000 )
# # model2.fit(X_train, y_train)

# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 5)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 5)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5 ]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True]

# param_grid = {
#     'bootstrap': bootstrap,
#     'max_depth': max_depth,
#     'max_features': max_features,
#     'min_samples_leaf': min_samples_leaf,
#     'min_samples_split': min_samples_split,
#     'n_estimators': n_estimators
# }
# # Create a based model
# rf = RandomForestRegressor()
# # Instantiate the grid search model
# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
#                           cv = 3, n_jobs = -1, verbose = 2)


# In[ ]:


# # Fit the grid search to the data
# grid_search.fit(X_train, y_train)


# In[ ]:


# grid_search.best_params_


# In[ ]:


# model2 = grid_search.best_estimator_
# results2 = model2.predict(X_test)
# score1 = metrics.mean_squared_error(y_test,results2 )
# score2 = metrics.r2_score(y_test,results2 )
# print('MSE: ',score1, '  R2 Score: ', score2)


# The above Gridsearch Algorthm has been executed in version 6 with following output. It took 281 min to get the best hyperparameters for RandomSearchRegression. 
# ```
# Fitting 3 folds for each of 360 candidates, totalling 1080 fits
# [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
# [Parallel(n_jobs=-1)]: Done  33 tasks      | elapsed:  7.2min
# /opt/conda/lib/python3.6/site-packages/joblib/externals/loky/process_executor.py:706: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.
#   "timeout or by a memory leak.", UserWarning
# [Parallel(n_jobs=-1)]: Done 154 tasks      | elapsed: 26.6min
# [Parallel(n_jobs=-1)]: Done 357 tasks      | elapsed: 79.6min
# [Parallel(n_jobs=-1)]: Done 640 tasks      | elapsed: 168.1min
# [Parallel(n_jobs=-1)]: Done 1005 tasks      | elapsed: 268.8min
# [Parallel(n_jobs=-1)]: Done 1080 out of 1080 | elapsed: 281.4min finished
# Out[31]:
# GridSearchCV(cv=3, error_score='raise-deprecating',
#              estimator=RandomForestRegressor(bootstrap=True, criterion='mse',
#                                              max_depth=None,
#                                              max_features='auto',
#                                              max_leaf_nodes=None,
#                                              min_impurity_decrease=0.0,
#                                              min_impurity_split=None,
#                                              min_samples_leaf=1,
#                                              min_samples_split=2,
#                                              min_weight_fraction_leaf=0.0,
#                                              n_estimators='warn', n_jobs=None,
#                                              oob_score=False, random_state=None,
#                                              verbose=0, warm_start=False),
#              iid='warn', n_jobs=-1,
#              param_grid={'bootstrap': [True],
#                          'max_depth': [10, 35, 60, 85, 110, None],
#                          'max_features': ['auto', 'sqrt'],
#                          'min_samples_leaf': [1, 2, 4],
#                          'min_samples_split': [2, 5],
#                          'n_estimators': [100, 575, 1050, 1525, 2000]},
#              pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
#              scoring=None, verbose=2)
# ```
# 
# Best parameters are given below:
# ```
# {'bootstrap': True,
#  'max_depth': 10,
#  'max_features': 'sqrt',
#  'min_samples_leaf': 4,
#  'min_samples_split': 5,
#  'n_estimators': 1525}
# ```
# 
# With above setup, the score was :
# ```
# MSE:  1104065.159504597   R2 Score:  0.0931234303865699
# ```

# With above parameters, let us try to build RandomForest model

# In[ ]:


model2 = RandomForestRegressor(bootstrap= True, max_depth = 10, 
 max_features = 'sqrt',
 min_samples_leaf =  10,
 min_samples_split = 8,
 n_estimators = 1550)

model2.fit(X_train, y_train)
results2 = model2.predict(X_test)
score1 = metrics.mean_squared_error(y_test,results2 )
score2 = metrics.r2_score(y_test,results2 )
print('MSE: ',score1, '  R2 Score: ', score2)


# We can see best parameters have  greatly improved the model. 

# ### LightGBM algorithm
# **LightGBM** is a gradient boosting framework that uses tree based learning algorithms. It is designed to be distributed and efficient with many advantages

# In[ ]:


import lightgbm as lgb
d_train = lgb.Dataset(X_train, label=y_train)
params = {}
params['learning_rate'] = 0.015
params['boosting_type'] = 'gbdt'
# params['boosting_type'] = 'dart'
params['objective'] = 'regression'
params['metric'] = 'mse'
params['sub_feature'] = 0.99
params['num_leaves'] = 10
params['min_data'] = 100
params['max_depth'] = 10000
y_train=y_train.ravel()
reg= lgb.train(params, d_train, 100)
results=reg.predict(X_test)
score1 = metrics.mean_squared_error(y_test,results )
score2 = metrics.r2_score(y_test,results )
print('MSE: ',score1, '  R2 Score: ', score2)


# Wow! Light GBM without much tuning we got much better results. 
# 
# **We can find best hyperparameters for LightGBM as improvement**

# Up next, let us try to create a GridSearchCV for LightGBM

# In[ ]:


# uniform(loc=0.2, scale=0.8)
estimator = lgb.LGBMRegressor(boosting_type= 'gbdt', metric='mse',objective='regression')


# In[ ]:


param_grid = {
    'learning_rate': [0.005, 0.01, 0.1, 0.5],
    'n_estimators': [int(x) for x in np.linspace(start = 20, stop = 2000, num = 7)],
    'num_leaves' : [int(x) for x in np.linspace(start = 10, stop = 100, num = 5)],
    'sub_feature' : [float(x) for x in np.linspace(start = 0.1, stop = 1, num = 3)]
}
gbm = GridSearchCV(estimator, param_grid, cv=3)


# In[ ]:


gbm.fit(X_train, y_train)
print('Best parameters found by grid search are:', gbm.best_params_)

gbm_best = gbm.best_estimator_
results = gbm_best.predict(X_test)
score1 = metrics.mean_squared_error(y_test,results )
score2 = metrics.r2_score(y_test,results )
print('MSE: ',score1, '  R2 Score: ', score2)


# ### Neural Networks
# Let us build a deep NN model for this regression

# In[ ]:


model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], init='uniform', activation='relu'))
model.add(Dense(64, init='uniform', activation='relu'))
model.add(Dense(64, init='uniform', activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()


# The model summary is shown above. Let us compile now

# In[ ]:


model.compile(loss='mse', optimizer='Adamax', metrics=['mse'])
history = model.fit(X_train, y_train, epochs=5000, batch_size=50,  verbose=0, validation_split=0.2)


# In[ ]:


results=model.predict(X_test)
score1 = metrics.mean_squared_error(y_test,results )
score2 = metrics.r2_score(y_test,results )
print('MSE: ',score1, '  R2 Score: ', score2)


# We see that the R2 score is around 0.77. 

# In[ ]:


print(history.history.keys())
# "Loss"
plt.subplots(1,1,figsize=(12,10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


# The above graph shows the improvement of loss with respect to epochs

# ### XGBoost
# Let us now try to implement XGBoost Algorithm
# 

# In[ ]:


xgb = xgboost.XGBRegressor(n_estimators=120, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=2)
xgb.fit(X_train,y_train)
results=xgb.predict(X_test)
score1 = metrics.mean_squared_error(y_test,results )
score2 = metrics.r2_score(y_test,results )
print('MSE: ',score1, '  R2 Score: ', score2)


# # Summary
# With all the above we can summarize our findings as below
# 
# |Model|Parameters|MSE |R2 Score|
# |-| -|-|-|
# |Linear Model (**Baseline**) |-|1157909|0.071 |
# |Polynomial Model| Degree = 2|1096834|0.092 | 
# |Random Forest|max_depth=10,max_features='sqrt',min_samples_leaf=10,min_samples_split=8, n_estimators=1550|1150355|0.077|
# |LightGBM|learning_rate= 0.15,boosting_type= 'gbdt,objective = 'regression',metric= 'mse',sub_feature= 0.3,num_leaves= 10,min_data = 1200,max_depth= 100|1094217|0.094|
# |Neural Net|4 layers= 9,64,32,32|1114359|0.078|
# |XGBoost| | | 0.077|
