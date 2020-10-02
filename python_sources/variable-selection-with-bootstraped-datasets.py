#!/usr/bin/env python
# coding: utf-8

# ### 0.0 Load modules

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

import collections
import itertools

import scipy.stats as stats
from scipy.stats import norm
from scipy.special import boxcox1p

import statsmodels
import statsmodels.api as sm
#print(statsmodels.__version__)

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
from plotly.subplots import make_subplots

init_notebook_mode(connected=True)

from tqdm import tqdm_notebook

from sklearn.preprocessing import scale, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, LinearRegression, ElasticNet,  HuberRegressor
from sklearn.metrics import mean_squared_error, balanced_accuracy_score, r2_score
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.utils import resample

from xgboost import XGBRegressor

#Model interpretation modules
import lime
import lime.lime_tabular
import shap
shap.initjs()

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ### 0.1 Load data

# In[ ]:


Combined_data = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
Combined_data.head()


# In[ ]:


print('Number of features: {}'.format(Combined_data.shape[1]))
print('Number of examples: {}'.format(Combined_data.shape[0]))


# In[ ]:


#for c in df.columns:
#    print(c, dtype(df_train[c]))
Combined_data.dtypes


# In[ ]:


Combined_data['last_review'] = pd.to_datetime(Combined_data['last_review'],infer_datetime_format=True) 


# # 1. Preprocessing and EDA

# ## 1.0 Missing data

# Many machine learning algorithms do badly when acting on inputs with missing data. To deal with this, we start by taking a count of missing values in each column.

# In[ ]:


total = Combined_data.isnull().sum().sort_values(ascending=False)
percent = (Combined_data.isnull().sum())/Combined_data.isnull().count().sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total','Percent'], sort=False).sort_values('Total', ascending=False)
missing_data.head(40)


# In[ ]:


Combined_data.drop(['host_name','name'], axis=1, inplace=True)


# In[ ]:


Combined_data[Combined_data['number_of_reviews']== 0.0].shape


# The NaN values in the last_review and reviews_per_month columns all occur for examples where no reviews were given in the first place. 
# 
# For reviews_per_month, I will fill those values with 0's.

# In[ ]:


Combined_data['reviews_per_month'] = Combined_data['reviews_per_month'].fillna(0)


# In[ ]:


earliest = min(Combined_data['last_review'])
Combined_data['last_review'] = Combined_data['last_review'].fillna(earliest)
Combined_data['last_review'] = Combined_data['last_review'].apply(lambda x: x.toordinal() - earliest.toordinal())


# In[ ]:


total = Combined_data.isnull().sum().sort_values(ascending=False)
percent = (Combined_data.isnull().sum())/Combined_data.isnull().count().sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total','Percent'], sort=False).sort_values('Total', ascending=False)
missing_data.head(40)


# ## 1.2 Choosing a prediction target [Smart Pricing Regressor]

# One of the machine learning models at AirBNB is Smart Pricing. After a client has entered the details of their rental, AirBNB suggests an appropriate price. The aim of this notebook is to build and train a Smart Pricing model for this dataset.

# ## 1.3 Price distribution

# I notice that the target has a highly skewed distribution. This can cause problems for machine learning algorithms such as linear regression. A log transformation and removal of outliers makes the distribution look much closer to normal.

# In[ ]:


fig, axes = plt.subplots(1,3, figsize=(21,6))
sns.distplot(Combined_data['price'], ax=axes[0])
sns.distplot(np.log1p(Combined_data['price']), ax=axes[1])
axes[1].set_xlabel('log(1+price)')
sm.qqplot(np.log1p(Combined_data['price']), stats.norm, fit=True, line='45', ax=axes[2]);


# In[ ]:


Combined_data = Combined_data[np.log1p(Combined_data['price']) < 8]
Combined_data = Combined_data[np.log1p(Combined_data['price']) > 3]


# In[ ]:


fig, axes = plt.subplots(1,3, figsize=(21,6))
sns.distplot(Combined_data['price'], ax=axes[0])
sns.distplot(np.log1p(Combined_data['price']), ax=axes[1])
axes[1].set_xlabel('log(1+price)')
sm.qqplot(np.log1p(Combined_data['price']), stats.norm, fit=True, line='45', ax=axes[2]);


# In[ ]:


Combined_data['price'] = np.log1p(Combined_data['price'])


# ## 1.4. Predictor distributions

# ### 1.4.0 A list of predictors

# In[ ]:


print(Combined_data.columns)


# ### 1.4.1 Host_id

# In[ ]:


print('In this dataset there are {} unique hosts renting out  a total number of {} properties.'.format(len(Combined_data['host_id'].unique()), Combined_data.shape[0]))


# Since the number of unique hosts is close to the total number of examples, we're not going to use hostname in our regression analysis since it would cause the number of parameters in our model to baloon! 
# 
# In real-life, when there is more data and perhaps some feature data on hosts, I expect past history of a host and of a property to be a strong predictor of price!

# In[ ]:


Combined_data = Combined_data.drop(['host_id', 'id'], axis=1)


# ### 1.4.2 Neighbourhood group

# I notice that Statten Island and the Bronx are highly underrepresented in this dataset. For Statten Island, the reason is that the population of the island is small. However, this can't be the case for the Bronx which has a population comparable (~1.4mln) to Manhattan (~1.6mln) or for for Brooklyn /Queens with their populations of ~2.5mln and ~2.4mln, respectively. 
# 
# This makes sense: Queens, the Bronx  and, to a fair extent Brooklyn, are residential neighborhoods unlike Manhattan which is a business centre as well as a tourist destination.

# In[ ]:


sns.catplot(x='neighbourhood_group', kind='count' ,data=Combined_data)
fig = plt.gcf()
fig.set_size_inches(12, 6)


# ### 1.4.3 Longitude and latitude

# Longitude and latitude are somewhat correlated with each other. This is because the locations of properties tend to come from clusters.

# In[ ]:


fig, axes = plt.subplots(1,3, figsize=(21,6))
sns.distplot(Combined_data['latitude'], ax=axes[0])
sns.distplot(Combined_data['longitude'], ax=axes[1])
sns.scatterplot(x= Combined_data['latitude'], y=Combined_data['longitude'])


# ### 1.4.4 Room type

# As far as room types, this dataset is balanced away from 'Shared room' properties. The proportions of private room and entire home/apt rentals are close, with entire home/apt dominating prive room by <10%.

# In[ ]:


sns.catplot(x='room_type', kind='count' ,data=Combined_data)
fig = plt.gcf()
fig.set_size_inches(8, 6)


# ### 1.4.5 Minimum nights

# In[ ]:


fig, axes = plt.subplots(1,2, figsize=(21, 6))

sns.distplot(Combined_data['minimum_nights'], rug=False, kde=False, color="green", ax = axes[0])
axes[0].set_yscale('log')
axes[0].set_xlabel('minimum stay [nights]')
axes[0].set_ylabel('count')

sns.distplot(np.log1p(Combined_data['minimum_nights']), rug=False, kde=False, color="green", ax = axes[1])
axes[1].set_yscale('log')
axes[1].set_xlabel('minimum stay [nights]')
axes[1].set_ylabel('count')


# In[ ]:


Combined_data['minimum_nights'] = np.log1p(Combined_data['minimum_nights'])


# ### 1.4.6 Reviews per month

# In[ ]:


fig, axes = plt.subplots(1,2,figsize=(18.5, 6))
sns.distplot(Combined_data[Combined_data['reviews_per_month'] < 17.5]['reviews_per_month'], rug=True, kde=False, color="green", ax=axes[0])
sns.distplot(np.sqrt(Combined_data[Combined_data['reviews_per_month'] < 17.5]['reviews_per_month']), rug=True, kde=False, color="green", ax=axes[1])
axes[1].set_xlabel('ln(reviews_per_month)')


# The distribution of the number of reviews per month is highly skewed however way we cut it. This is because there is a large weight on small numbers: there are a lot of properties which only get a few reviews and a rather fat tail of properties which get a lot of reviews. 
# 
# One explanation would be that the properties which are available a larger fraction of the year get more reviews. However, a scatter plot of reviews_per_month and availability_365 variables shows no evidence of a relationship so that explanation would appear to not be valid.

# In[ ]:


fig, axes = plt.subplots(1,1, figsize=(21,6))
sns.scatterplot(x= Combined_data['availability_365'], y=Combined_data['reviews_per_month'])


# In[ ]:


Combined_data['reviews_per_month'] = Combined_data[Combined_data['reviews_per_month'] < 17.5]['reviews_per_month']


# ### 1.4.7 Availability_365

# This distribution is highly skewed towards the low and high end. The dataset contains a hiuge number of properties that are available only for a couple of days each year, and a decent number that are available for > 300 days. 

# In[ ]:


fig, axes = plt.subplots(1,1,figsize=(18.5, 6))
sns.distplot(Combined_data['availability_365'], rug=False, kde=False, color="blue", ax=axes)
axes.set_xlabel('availability_365')
axes.set_xlim(0, 365)


# ## 1.5 Bivariate correlations

# ### 1.5.0 Pearson correlation matrix

# In[ ]:


corrmatrix = Combined_data.corr()
f, ax = plt.subplots(figsize=(15,12))
sns.heatmap(corrmatrix, vmax=0.8, square=True)


# There don't appear to exist obvious, strong correlations between these variables. 
# 
# However, the number of reviews per month is fairly (40%) correlated with the total number of reviews and the the total number of reviews is correlated (at 30%) with the availability of the property. Both of these correlations make sense.
# 
# It's also interesting that the longitude is anticorrelated (at 20%) with the price. That also makes sense - property in the Bronx and in Queens is cheaper than Manhattan and Brooklyn.

# ### 1.5.1 PairPlot

# In[ ]:


#sns.pairplot(Combined_data.select_dtypes(exclude=['object']))#


# ## 1.6 Encoding categorical features

# In[ ]:


categorical_features = Combined_data.select_dtypes(include=['object'])
print('Categorical features: {}'.format(categorical_features.shape))


# In[ ]:


categorical_features_one_hot = pd.get_dummies(categorical_features)
categorical_features_one_hot.head()


# In[ ]:


Combined_data['reviews_per_month'] = Combined_data['reviews_per_month'].fillna(0)


# ## 1.7 Save transformed dataframe for future use

# In[ ]:


numerical_features =  Combined_data.select_dtypes(exclude=['object'])
y = numerical_features.price
numerical_features = numerical_features.drop(['price'], axis=1)
print('Numerical features: {}'.format(numerical_features.shape))


# In[ ]:


X = np.concatenate((numerical_features, categorical_features_one_hot), axis=1)
X_df = pd.concat([numerical_features, categorical_features_one_hot], axis=1)
#print('Dimensions of the design matrix: {}'.format(X.shape))
#print('Dimension of the target vector: {}'.format(y.shape))


# In[ ]:


Processed_data = pd.concat([X_df, y], axis = 1)
#Processed_data.to_csv('NYC_Airbnb_Processed.dat')


# ## 1.8 Train-test split

# I'm going to split the data into a test set and a training set. I will hold out the test set until the very end and use the error on those data as an unbiased estimate of how my models did. 
# 
# I might perform a further split later on the training set into training set proper and a validation set or I might cross-validate.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


print('Dimensions of the training feature matrix: {}'.format(X_train.shape))
print('Dimensions of the training target vector: {}'.format(y_train.shape))
print('Dimensions of the test feature matrix: {}'.format(X_test.shape))
print('Dimensions of the test target vector: {}'.format(y_test.shape))


# ## 1.9 Rescale the design matrix

# I now scale the design matrix with sklearn's RobustScaler() so that each predictor has zero mean and unit variance. This helps the convergence of machine learning algorithms such as linear regression.
# 
# I avoid data snooping by defining the scaleing transformation based on the training data not the test data.

# In[ ]:


scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# In[ ]:


Xy_train = pd.concat([pd.DataFrame(X_train, columns = X_df.columns), pd.DataFrame(y_train, columns=['price'])], axis=1)


# In[ ]:


Xy_train.shape


# # 2. Models

# ## 2.1 Cross-validation routine

# I will score models based on K-fold cross-validation with 5 folds.

# In[ ]:


n_folds = 5

# squared_loss
def rmse_cv(model, X = X_train, y=y_train):
    kf = KFold(n_folds, shuffle=True, random_state = 91).get_n_splits(numerical_features)
    return cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kf)


# ## 2.3 Lasso Regression

# ### Original data

# I find the best value of the L2 penalty hyperparameter with a telescopic search based on cross-validation scores. I then train the Ridge model on the entire training set and test how it performs on the held-out test set.

# In[ ]:


best_alpha = 0.0001
lasso_CV_best = -rmse_cv(Ridge(alpha = best_alpha))
lasso = Lasso(alpha = best_alpha) 
lasso.fit(X_train, y_train) 
y_train_lasso = lasso.predict(X_train)
y_test_lasso = lasso.predict(X_test)
lasso_results = pd.DataFrame({'sample':['original'],
            'CV error': lasso_CV_best.mean(), 
            'CV std': lasso_CV_best.std(),
            'training error': [mean_squared_error(y_train_lasso, y_train)],
            'test error': [mean_squared_error(y_test_lasso, y_test)],
            'training_r2_score': [r2_score(y_train, y_train_lasso)],
            'test_r2_score': [r2_score(y_test, y_test_lasso)]})
lasso_results


# In[ ]:


lasso_results


# In[ ]:


lasso_coefs = pd.DataFrame(lasso.coef_, columns=['original'], index = X_df.columns)
lasso_coefs


# ### Bootstrapped data

# In[ ]:



lasso_all = lasso_results
lasso_coefs_all = lasso_coefs

for i in range(0, 100):
    print('Bootstrap sample no. '+str(i))
    X_bs, y_bs = resample(X_train, y_train, n_samples = X_df.shape[0], random_state=i) 
    best_alpha = 0.0001
    lasso_CV_best = -rmse_cv(Lasso(alpha = best_alpha))
    lasso = Lasso(alpha = best_alpha) 
    lasso.fit(X_bs, y_bs) 
    y_bs_lasso = lasso.predict(X_bs)
    y_test_lasso = lasso.predict(X_test)
    lasso_bs_results = pd.DataFrame({'sample':['Bootstrap '+str(i)],
            'CV error': lasso_CV_best.mean(), 
            'CV std': lasso_CV_best.std(),
            'training error': [mean_squared_error(y_bs, y_bs_lasso)],
            'test error': [mean_squared_error(y_test_lasso, y_test)],
            'training_r2_score': [r2_score(y_bs, y_bs_lasso)],
            'test_r2_score': [r2_score(y_test, y_test_lasso)]})
    lasso_all = pd.concat([lasso_all, lasso_bs_results], ignore_index=True)
    lasso_coefs_all = pd.concat([lasso_coefs_all, pd.DataFrame(lasso.coef_, columns=['Bootstrap' + str(i)], index = X_df.columns)], axis=1)


# In[ ]:


lasso_all


# In[ ]:


lasso_all.describe()


# In[ ]:


lasso_coefs_all.head(10)


# In[ ]:


lasso_coefs_all['mean'] = lasso_coefs.mean(axis=1)
lasso_coefs_all.sort_values(by='mean', inplace=True)
lasso_coefs_all


# In[ ]:


lasso_coefs_all['nonzero_vals'] = lasso_coefs_all.astype(bool).sum(axis=1)
lasso_coefs_all.sort_values(by='nonzero_vals', inplace=True)
lasso_coefs_all


# In[ ]:


sns.distplot(lasso_coefs_all['nonzero_vals'], kde=False)


# In[ ]:


trace = go.Histogram(
    x=lasso_coefs_all['nonzero_vals'],
    marker=dict(
        color='blue'
    ),
    opacity=0.75
)

layout = go.Layout(
    title='LASSO Variable Importance on boot',
    height=450,
    width=1200,
    xaxis=dict(
        title='No. nonzero occurrences'
    ),
    yaxis=dict(
        title='No. features'
    ),
    bargap=0.2,
)

data= [trace]

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)


# In[ ]:


#RR_coefs_all[RR_coefs_all.index=='longitude']
lasso_coefs_all.loc['longitude']


# In[ ]:


def getACoefficientSlice(start=0, end=1):
    df_lasso = lasso_coefs_all.iloc[start:end]
    df_lasso['std'] = df_lasso.std(axis=1)
    df_lasso = df_lasso.sort_values(by='std', ascending=True)
    df_lasso.drop(columns=['mean','std'], inplace=True)
    return df_lasso

getACoefficientSlice(0,10)


# In[ ]:


fig = go.Figure()

df = getACoefficientSlice(0,10)

for predictor in df.index:
    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))
    
fig.update_layout(showlegend=False)
fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))

fig.show()


# In[ ]:


fig = go.Figure()

df = getACoefficientSlice(10,20)

for predictor in df.index:
    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))
    
fig.update_layout(showlegend=False)
fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))

fig.show()


# In[ ]:


fig = go.Figure()

df = getACoefficientSlice(20,30)

for predictor in df.index:
    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))
    
fig.update_layout(showlegend=False)
fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))

fig.show()


# In[ ]:


fig = go.Figure()

df = getACoefficientSlice(30,40)

for predictor in df.index:
    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))
    
fig.update_layout(showlegend=False)
fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))

fig.show()


# In[ ]:


fig = go.Figure()

df = getACoefficientSlice(40,50)

for predictor in df.index:
    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))
    
fig.update_layout(showlegend=False)
fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))

fig.show()


# In[ ]:


fig = go.Figure()

df = getACoefficientSlice(50,60)

for predictor in df.index:
    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))
    
fig.update_layout(showlegend=False)
fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))

fig.show()


# In[ ]:


fig = go.Figure()

df = getACoefficientSlice(60,70)

for predictor in df.index:
    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))
    
fig.update_layout(showlegend=False)
fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))

fig.show()


# In[ ]:


fig = go.Figure()

df = getACoefficientSlice(70,80)

for predictor in df.index:
    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))
    
fig.update_layout(showlegend=False)
fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))

fig.show()


# In[ ]:


fig = go.Figure()

df = getACoefficientSlice(80,90)

for predictor in df.index:
    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))
    
fig.update_layout(showlegend=False)
fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))

fig.show()


# In[ ]:


fig = go.Figure()

df = getACoefficientSlice(90,100)

for predictor in df.index:
    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))
    
fig.update_layout(showlegend=False)
fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))

fig.show()


# In[ ]:


fig = go.Figure()

df = getACoefficientSlice(100,110)

for predictor in df.index:
    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))
    
fig.update_layout(showlegend=False)
fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))

fig.show()


# In[ ]:


fig = go.Figure()

df = getACoefficientSlice(110,120)

for predictor in df.index:
    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))
    
fig.update_layout(showlegend=False)
fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))

fig.show()


# In[ ]:


fig = go.Figure()

df = getACoefficientSlice(120,130)

for predictor in df.index:
    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))
    
fig.update_layout(showlegend=False)
fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))

fig.show()


# In[ ]:


fig = go.Figure()

df = getACoefficientSlice(130,140)

for predictor in df.index:
    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))
    
fig.update_layout(showlegend=False)
fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))

fig.show()


# In[ ]:


fig = go.Figure()

df = getACoefficientSlice(140,150)

for predictor in df.index:
    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))
    
fig.update_layout(showlegend=False)
fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))

fig.show()


# In[ ]:


fig = go.Figure()

df = getACoefficientSlice(160,170)

for predictor in df.index:
    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))
    
fig.update_layout(showlegend=False)
fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))

fig.show()


# In[ ]:


fig = go.Figure()

df = getACoefficientSlice(170,180)

for predictor in df.index:
    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))
    
fig.update_layout(showlegend=False)
fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))

fig.show()


# In[ ]:


fig = go.Figure()

df = getACoefficientSlice(180,190)

for predictor in df.index:
    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))
    
fig.update_layout(showlegend=False)
fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))

fig.show()


# In[ ]:


fig = go.Figure()

df = getACoefficientSlice(190,200)

for predictor in df.index:
    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))
    
fig.update_layout(showlegend=False)
fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))

fig.show()


# In[ ]:


fig = go.Figure()

df = getACoefficientSlice(200,210)

for predictor in df.index:
    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))
    
fig.update_layout(showlegend=False)
fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))

fig.show()


# In[ ]:


fig = go.Figure()

df = getACoefficientSlice(210,220)

for predictor in df.index:
    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))
    
fig.update_layout(showlegend=False)
fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))

fig.show()


# In[ ]:


fig = go.Figure()

df = getACoefficientSlice(220,230)

for predictor in df.index:
    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))
    
fig.update_layout(showlegend=False)
fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))

fig.show()


# In[ ]:


fig = go.Figure()

df = getACoefficientSlice(230,243)

for predictor in df.index:
    fig.add_trace(go.Box(y=df.loc[predictor], name=predictor, boxpoints='outliers'))
    
fig.update_layout(showlegend=False)
fig.update_xaxes(title_font=dict(size=28, family='Courier', color='crimson'))

fig.show()


# In[ ]:


lasso_coefs_all.drop(columns=['mean'], inplace=True)
lasso_coefs_all.to_csv('Lasso_regr')


# In[ ]:




