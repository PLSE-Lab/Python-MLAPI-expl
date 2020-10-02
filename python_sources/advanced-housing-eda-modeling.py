#!/usr/bin/env python
# coding: utf-8

# # Housing Regression in Python
# 

# In[ ]:


# library import
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
import warnings 
from scipy.stats import norm

warnings.simplefilter(action='ignore', category=FutureWarning)

def coef_plot(model, X, abs_ = False, n = 5, figsize = (12, 8)):
    if abs_ == False:
        coefs = pd.DataFrame({'name': X.columns, 'coef' : model.coef_}).sort_values('coef', ascending = False)
    else:
        coefs = pd.DataFrame({'name': X.columns, 'coef' : np.abs(model.coef_)}).sort_values('coef', ascending = False)
    plt.figure(figsize = figsize)
    sns.pointplot(y="name", x="coef",
                  data=coefs.head(n), ci=None, color = 'C0')
    sns.barplot(y = "name", x= "coef", data=coefs.head(n), ci=None, color = 'C0', alpha = 0.2)
    plt.title('Coeficient Plot')
    plt.tight_layout()


# ### Data import

# In[ ]:


# Use pandas to read in CSV files
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# row and column number in train set 
print("train rows amd columns", train.shape)
print("test rows amd columns", test.shape)


# ### First steps: 
# 
# ### 1) Look at variable distributions 
# 
# ### 2) look at correlations with our outcome variable
# 

# ### Histogram of all numeric variables:

# In[ ]:


# plot histograms of all the numeric columns
train.hist(figsize = (12, 7), bins = 40)
plt.tight_layout()
plt.show()


# * A lot of skewed / non-normal distributions here!
# 
# * Skewed vairables can be a problem for some learners, particularly linear regression

# In[ ]:


plt.figure(figsize = (12, 6))
train.skew().sort_values().plot(kind = 'bar', color = 'C0')
plt.title('Skew of Variables')


# ### Distribution of Our Outcome or y variable

# In[ ]:


fig, ax = plt.subplots(2, 1, figsize = (12, 8)) 

sns.distplot(train['SalePrice'], fit = norm, ax = ax[0], label = 'emperical distribution')
sns.distplot(train['SalePrice'], fit = norm, label = 'gaussian fit', hist = False, kde = False, ax = ax[0])
ax[0].legend()
sns.distplot(np.log(train['SalePrice']), ax = ax[1], label = 'emerpical distribution')
sns.distplot(np.log(train['SalePrice']), fit = norm, label = 'gaussian fit', hist = False, kde = False, ax = ax[1])
plt.legend()
plt.show()


# * Outcome somewhat right skewed
# 
# * A log transformation might help with this

# ### Visualize Correlation between numeric X variables and Outcome

# In[ ]:


cormat = train.corr(method='spearman')
ax = sns.clustermap(cormat, center = np.median(cormat))
plt.show()


# * There is a high correlation between many of our predictors, this can be a problem for some learners, particularly linear regression
# 
# * High correlation between variables causes multi-colinearity
# 

# In[ ]:


plt.figure(figsize = (12, 8))
cormat = train.corr(method = 'spearman')[['SalePrice']].sort_values(by = 'SalePrice', ascending = True)[:-1]
sns.heatmap(cormat, annot = True, center = np.median(cormat))
plt.title("Numeric Variable's Correaltion with Sale Price")
plt.show()


# * OverallQual and GrLivArea have the highest absolute correlation with our outcome variables
# 

# ### Categorical variables
# 
# * When we one-hot encode a column, we create a column for each unique value of the column
# 
# * If we use the describe function with just object type columns we can get some summary stats

# In[ ]:


train.select_dtypes('object').describe()


# * Some of these variables have a lot of unique values, I should probably get rid of some of the unique values with few observations in the columns that are high cardinaltiy (have a lot of unique values), especially the Neighborhood column.

# In[ ]:


plt.figure(figsize = (12, 6))
train['Neighborhood'].value_counts().plot(kind = 'barh', color = 'C0')
plt.xlabel('Count')
plt.title("Woah! That's a Lot of Neighborhoods")
sns.despine()
plt.show()


# lets get rid of some of these:

# In[ ]:


for col in train.select_dtypes('object').columns:
    top = train[col].value_counts().head(10)
    train[col] = [x if x in top else "other" for x in train[col]]
    test[col] = [x if x in top else "other" for x in test[col]]


# ### Examine Missing Values

# Note: We won't have any missing values in the categorical columns they will have already been replaced with the string "other". I have found in practice this actually works quite well and solves a lot of problems that you can run into with categorical variables (especially in production).

# In[ ]:


train.isna().sum()[train.isna().sum() > 0].sort_values().plot(kind = 'barh', color = 'C0', figsize = (10, 4))
plt.title('Missing Values')
plt.show()


# ### Basic Models
# 
# We want to use our training data as efficiently as possible when fitting models.  To do that we will use a process called cross-validation.  
# 
# I see many people reccomend to start off with OLS linear regression when modeling.  I disagree with this idea, especially when you already know that there is high correlation between input variables.  Lasso and ridge regression create linear models, similar to OLS regression, but with ridge and lasso regression we shrink coefficients to deal with a problem that arises in OLS regerssion called multi-colinearity.

# We need to one hot econde our categorical variables. There is a problem that we have to deal with. If we onehot encode the two datasets seperately, the test set will have less columns, because some of the unique categories in the train set don't show up in the test set. The easiest way to deal with this is to put them together to onehot encode the matrix.  This does have the potential to introduce leakage into some problems (I'd argue it doesn't here).  

# In[ ]:


print(pd.get_dummies(train).shape)
print(pd.get_dummies(test).shape)


# In the chunk below I create dummies for the train and test set and define the X matrix and y vector.

# In[ ]:


from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer

y = train['SalePrice']
X = train.drop(['SalePrice'], 1)

train_objs_num = len(X)
dataset = pd.concat([X, test])
dataset = pd.get_dummies(dataset)
X = dataset[:train_objs_num].copy()
test = dataset[train_objs_num:].copy()


# Sklearn does have a LassoCV object that performs quick easy crossvalidation to select a top alpha value. Unfortunatley unless you use loo cross validation, it doesn't save the scores. Because of this, I ussually tune the alpha parameter for ridge regerssion with a loop like this. This way I can make a plot of how the model metric changes with different values of lambda

# ## Lasso Regression

# In[ ]:


import warnings 
from sklearn.exceptions import ConvergenceWarning


# these warnings are telling me that some of the alpha values are too low, I know this. I am attempting to do a wide grid search.
warnings.filterwarnings(action='ignore', category=ConvergenceWarning,)

alphas = []
scores = []

for i in [1e-6, 1e-4, 1e-2,.1, 1, 5, 10, 20, 30, 50, 100,200,250, 300,400,500,750,1000]:
    
    pipe = Pipeline([
                 ('Imputer', SimpleImputer(strategy = 'most_frequent')),
                 ('scaler', RobustScaler()),
                ('lasso', Lasso(alpha= i, max_iter= 10000))
            ])
    pipe.fit(X, y)
    score = cross_val_score(pipe, X, y, cv = 5)
    # nested loops in python are really ugly and should be avoided by I'm lazy
    for x in score:
        scores.append(x)
        alphas.append(i)
ridge_frame = pd.DataFrame({'alpha': alphas, 'score': scores})

top_alpha = ((ridge_frame.groupby('alpha', as_index=False).mean().sort_values('score')))

top_alpha = top_alpha.iloc[-1:, 0].values
print(top_alpha[0])


# It seems that an alpha of 200 produces the highest cross validation accuracy

# In[ ]:


plt.figure(figsize = (12, 8))
sns.lineplot('alpha', 'score', data = ridge_frame, ci = "sd")
plt.title('Crossvalidated Alpha vs R^2 Score + or - 1 sd')
plt.axvline(top_alpha, color = "C1")
plt.xscale('log')
plt.show()


# In order to figure out the most important variables, I can take the absolute value of the coefficients.

# In[ ]:


pipe = Pipeline([
                  ('Imputer', SimpleImputer(strategy = 'most_frequent')),  
                 ('scaler', RobustScaler()),
                ('lasso', Lasso(alpha= top_alpha))
            ])
pipe.fit(X, y)
score = cross_val_score(pipe, X, y, cv = 5)

rcv = pipe.named_steps['lasso']

coef_plot(rcv, X, abs_ = True, n = 40)
print(np.mean(scores))
print(np.std(scores))
plt.show()


# In[ ]:


from sklearn.linear_model import Ridge
alphas = []
scores = []

for i in [1e-6, 1e-4, 1e-2,.1, 1, 5, 10, 20, 30, 50, 100,200,250, 300,400,500,750, 1000]:
    
    pipe = Pipeline([
                 ('Imputer', SimpleImputer(strategy = 'most_frequent')),
                 ('scaler', RobustScaler()),
                ('ridge', Ridge(alpha= i, max_iter= 10000))
            ])
    pipe.fit(X, y)
    score = cross_val_score(pipe, X, y, cv = 5)
    # nested loops in python are really ugly and should be avoided by I'm lazy
    for x in score:
        scores.append(x)
        alphas.append(i)
ridge_frame = pd.DataFrame({'alpha': alphas, 'score': scores})

top_alpha = ((ridge_frame.groupby('alpha', as_index=False).mean().sort_values('score')))

top_alpha = top_alpha.iloc[-1:, 0].values


# In[ ]:


pipe = Pipeline([
                  ('Imputer', SimpleImputer(strategy = 'most_frequent')),  
                 ('scaler', RobustScaler()),
                ('ridge', Ridge(alpha= top_alpha))
            ])
pipe.fit(X, y)
score = cross_val_score(pipe, X, y, cv = 5)

rcv = pipe.named_steps['ridge']

coef_plot(rcv, X, abs_ = True, n = 50)
print(np.mean(scores))
print(np.std(scores))
plt.show()


# The lasso regression outperforms the ridge, which tells me that some feature selection may be necessary. (Lasso shrinks some coefficients to 0)

# Lets try a random forest regressor to compare performance

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV



pipe = Pipeline([('imp', SimpleImputer(strategy = 'most_frequent')),
                 ('scaler', RobustScaler()),
                ('rf', RandomForestRegressor())
            ])
pipe.fit(X, y)
params_rf = {'rf__n_estimators' : [100, 350, 500, 1000, 2000],
            'rf__max_features': ['log2', 'auto', 'sqrt'],
            'rf__min_samples_leaf': [1, 2, 5, 10, 30],
            "rf__min_samples_split": [2, 3, 5, 7,9,11,13,15,17],
            "rf__max_depth" : [None, 1, 3, 5, 7, 9, 11]}

# Import GridSearchCV

# Instantiate grid_rf
grid_rf = RandomizedSearchCV(estimator=pipe,
                       param_distributions=params_rf,
                       cv=5,
                       verbose=0,
                       n_jobs=-1,
                       n_iter = 30)

grid_rf.fit(X,y)
bestmodel = grid_rf.best_estimator_

print(grid_rf.best_score_)


# The random forest performs a bit better than the linear models which tells me that the underlying relationship may be non-linear.  Lets try another non-linear estimator

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_log_error

pipe = Pipeline([('imp', SimpleImputer(strategy = 'most_frequent')),
                 ('scaler', RobustScaler()),
                ('gbm', GradientBoostingRegressor())
            ])

pipe.fit(X, y)
params_gbm = {'gbm__n_estimators' : [100, 350, 500, 1000, 2000],
            'gbm__max_features': ['log2', 'auto', 'sqrt'],
            'gbm__min_samples_leaf': [1, 2, 10, 30],
            "gbm__min_samples_split": [2, 3,5,7,9,11],
            "gbm__learning_rate" : [1e-5,1e-4, 1e-3, 1e-2, 0.1, 1]}
# Import GridSearchCV

# Instantiate grid_rf
grid_rf = RandomizedSearchCV(estimator=pipe,
                       param_distributions=params_gbm,
                       cv=5,
                       verbose=0,
                       n_jobs=-1, 
                       n_iter = 30)

grid_rf.fit(X,y)
bestmodel = grid_rf.best_estimator_
print(grid_rf.best_score_)

