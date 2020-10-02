#!/usr/bin/env python
# coding: utf-8

# ## Not really an intro from me...

# This is my first kernel on Kaggle and in this kernel I would like to to go over the entire analytic workflow. We all know that data cleaning is the dirtist part of the whole project but it is also the foundation for further analyses and a necessary process to get to understand your data. So you may see that the data cleaning part occupies most of the space. As for the techqniues, I am just throwing out a brick to attract jade ^ ^. There are of course far more techniques that can be incorporated and more precise analyses can be done. I only point out the basic steps I think are most essential. Now let us start!

# <img src="https://rentguard.co.uk/wp-content/uploads/2019/02/thinkstockphotos-682308948.jpg" width="625px">

# Roadmap:
# 1. Import data
# 1. Preliminary analysis
#     * For numerical variables: descriptive stats, distribution and normality test, correlation analysis
#     * For categorical variables: frequency table, correlation analysis 
# 1. Data preparation
#     * Missing data imputation
#     * For numerical variables: standardization or Yeo-Johnson transformation
#     * For categorical varaibles: one-hot encoding
#     * Principal component analysis
# 1. Modeling
#     * Random Forests
#     * XGBoost
#     * Neural Networks

# ## Import Data

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context('notebook', font_scale=1.5)
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.head()


# In[ ]:


train.shape


# In[ ]:


test.head()


# In[ ]:


test.shape


# ## Preliminary Analysis

# ### Numeric Variables

# I split the analysis based on the different types of variables. It is easier to describe the numeric variables since we can calculate descriptive statistics and apply various statistical tests.

# In[ ]:


train_num = train.select_dtypes(exclude=['object']).columns
train_num


# #### 1. Basic statistics

# In[ ]:


train.describe()


# The first line "count" indicates several variables that have missing values given the total number of observations is 1460, - LotFrontage, MasVnrArea, GarageYrBlt. LotFrontage's missing values are the most, which is 259. Judged from mean, quantiles, etc., we can also identify many skewed variables, for example, BsmtFinSF1 (Type 1 finished square feet), BsmtFullBath (Basement full bathrooms), LotArea (Lot size in square feet), MasVnrArea (Masonry veneer area in square feet).

# #### 2. Check the distribution

# The basic idea is to observe the distribution of variables in plots. I also apply shapiro wilk normality test here for reference. But one thing I find about shapiro test is that it is not so reliable when the data is large. It can be easily influenced by even small deviations and since its null hypothesis is normality assumption, it tends to reject the null in fact. Check more explanations [here](https://stats.stackexchange.com/questions/2492/is-normality-testing-essentially-useless?noredirect=1&lq=1).

# In[ ]:


from scipy.stats import shapiro
# apply shapiro test
stat, p = shapiro(train['BsmtFinSF1'])
print('Skewness=%.3f' %train['BsmtFinSF1'].skew())
print('Statistics=%.3f, p=%.3f' %(stat, p))

# set alpha to 0.05: when p > 0.05, accept the null hypothesis; when p < 0.05, reject the null
alpha = 0.05
if p > alpha:
    print('Data looks normal (fail to reject H0)')
else:
    print('Data does not look normal (reject H0)')

sns.distplot(train['BsmtFinSF1']);


# **BsmtFinSF1** is highly skewed and its distribution is not normal based on the Shapiro-Wilk test.

# In[ ]:


stat, p = shapiro(train['BsmtFullBath'])
print('Skewness=%.3f' %train['BsmtFullBath'].skew())
print('Statistics=%.3f, p=%.3f' %(stat, p))

sns.distplot(train['BsmtFullBath']);


# **BsmtFullBath** is moderately skewed but its distribution is not normal either because p value is smaller than 0.05. Acutally we can observe its distribution in the plot which has two peaks.

# In[ ]:


stat, p = shapiro(train['LotArea'])
print('Skewness=%.3f' %train['LotArea'].skew())
print('Statistics=%.3f, p=%.3f' %(stat, p))

sns.distplot(train['LotArea']);


# Another highly skewed variables **LotArea** has a long left tail. The majority of values are around 10,000 but it maximum value can reach to over 200,000.

# In[ ]:


stat, p = shapiro(np.log(train['LotArea']))
print('After log transformation...')
print('Skewness=%.3f' %np.log(train['LotArea']).skew())
print('Statistics=%.3f, p=%.3f' %(stat, p))

sns.distplot(np.log(train['LotArea']));


# After we apply the log transformation to LotArea, the skewness is reduced a lot but it is still not a normal distribution based on the test.

# In[ ]:


stat, p = shapiro(train['MasVnrArea'].dropna())
print('Skewness=%.3f' %train['MasVnrArea'].skew())
print('Statistics=%.3f, p=%.3f' %(stat, p))

sns.distplot(train['MasVnrArea'].dropna());


# In[ ]:


masvnrarea_std = (train['MasVnrArea'] - np.mean(train['MasVnrArea'])) / np.std(train['MasVnrArea'])
stat, p = shapiro(masvnrarea_std.dropna())
print('Skewness=%.3f' %masvnrarea_std.skew())
print('Statistics=%.3f, p=%.3f' %(stat, p))

sns.distplot(masvnrarea_std.dropna());


# Try the standardization with **MasVnrArea**, but it does not change anything. 

# **Now let us take a look at our response variable: SalePrice**

# In[ ]:


stat, p = shapiro(train['SalePrice'])
print('Skewness=%.3f' %train['SalePrice'].skew())
print('Statistics=%.3f, p=%.3f' %(stat, p))

sns.distplot(train['SalePrice']);


# In[ ]:


stat, p = shapiro(np.log(train['SalePrice']))
print('After log transformation...')
print('Skewness=%.3f' %np.log(train['SalePrice']).skew())
print('Statistics=%.3f, p=%.3f' %(stat, p))

sns.distplot(np.log(train['SalePrice']));


# Before log transformation, SalePrice is highly skewed. While log transformation can help with reducing skewness, it can not transform the data to normal distribution (perhaps). But at least from the plot, we can see that it is improved a lot compared to before. So I will still go with the log transformation.

# #### 3. Correlation analysis

# In[ ]:


train_num_corr = train[train_num].drop(['Id'], axis=1)
corr = pd.DataFrame(train_num_corr.corr(method = 'pearson')['SalePrice'])
corr.sort_values(['SalePrice'], ascending= False)


# OverallQual (Overall material and finish quality), GrLivArea (Above grade (ground) living area square feet), GarageCars (Size of garage in car capacity), GarageArea (Size of garage in square feet), TotalBsmtSF (Total square feet of basement area), 1stFlrSF (First Floor square feet), etc. are highly correlated with our response variable, SalePrice. We can then conclude that area-related variables (including ground living area, garage area, basement area, number of rooms) as well as quality and year (like construction year, remodeled year) are the major continous factors associated with the sale price of a house.

# In[ ]:


cmap = sns.cubehelix_palette(light = 0.95, as_cmap = True)
sns.set(font_scale=1.2)
plt.figure(figsize = (9, 9))
sns.heatmap(abs(train_num_corr.corr(method = 'pearson')), vmin = 0, vmax = 1, square = True, cmap = cmap);


# From the last row of the heatmap, we can get the same information as from the previous correlation table. Other than this, we can also find the multicollinearity problems with the data, for example, GarageCars and GarageArea, YearBuilt and GarageYrBlt.

# ### Categorical Variables

# #### 1. Check the distribution

# For categorical variables, I make a frequency table to show the distribution.

# In[ ]:


train_cat = train.select_dtypes(include=['object']).columns
train_cat


# In[ ]:


pd.set_option('display.max_rows', 300)
df_output = pd.DataFrame()
# loop through categorical variables, and append calculated stats together
for i in range(len(train_cat)):
    c = train_cat[i]
    df = pd.DataFrame({'Variable':[c]*len(train[c].unique()),
                       'Level':train[c].unique(),
                       'Count':train[c].value_counts(dropna = False)})
    df['Percentage'] = 100 * df['Count']  / df['Count'].sum()
    df_output = df_output.append(df, ignore_index = True)
    
df_output


# Judged from the frequency table, we can infer that the majority of categorical variables are unbalanced. Several classes even only have 1 or 2 observations. Worsely, Alley, FireplaceQu, PoolQC, Fence, MiscFeature contain missing values over 50%.

# #### 2. Correlation with SalePrice

# Take several variables for example:

# In[ ]:


sns.set(style = 'whitegrid', rc = {'figure.figsize':(10,7), 'axes.labelsize':12})
sns.boxplot(x = 'MSZoning', y = 'SalePrice', palette = 'Set2', data = train, linewidth = 1.5);


# **MSZoning**. Most of houses are in the Residential Low Density (79%) and Residential Medium Density (15%) areas. However, there is a large variance within the Residential Low (RL) zone. Floating Village(FV) is a special area where a retirement community was developed and have the highest median price among the all. But there are only 16 observations falling within this category and their prices do not seem quite stable looking at the shape of the box of FV.

# In[ ]:


col_order = train.groupby(['Neighborhood'])['SalePrice'].aggregate(np.median).reset_index().sort_values('SalePrice')
p = sns.boxplot(x = 'Neighborhood', y = 'SalePrice', palette = 'Set2', data = train, order=col_order['Neighborhood'], linewidth = 1.5)
plt.setp(p.get_xticklabels(), rotation=45);


# **Neighborhood**. Besides the zoning classfication, neighborhood also makes a difference. Houses located at Northridge Heights (NridgHt) have higher sale prices than those in other areas generally but the variance is large. The difference between median price of MeadowV (neighborhood with the lowest house prices) and that of NridgHt is over $200,000.

# In[ ]:


sns.boxplot(x = 'HouseStyle', y = 'SalePrice', palette = 'Set2', data = train, linewidth = 1.5);


# **HouseStyle**. Locations matter a lot when considering house prices, then what about the characteristics of house itself? Popular house styles are 2 Story (50%) and 1 Story (30%). 2 story and 2.5 story (2nd level finished) houses can be sold at relatively higher prices, around 200,000 dollars, while the prices of 1.5 story (2nd level unfinished) houses are mostly around 110,000 dollars. Notably, for multiple story houses, 2 level finished or unfinished have an obvious relationship with house prices.

# In[ ]:


sns.scatterplot(x = 'YearBuilt', y = 'SalePrice', data = train, hue = 'HouseStyle', style = 'HouseStyle', palette = 'colorblind');


# After introducing year of built into our plot, we can see that houses built in recent years tend to have higher sale prices, especially since 1960. YearBuilt also has certain association with HouseStyle. We can identify at least three clusters in this plot, blue points (2Story) mostly near the right, yellow (1Story) most in the between, and green (1.5Fin) on the left. 
# 
# Intuitively, we assume that the price of house with more stories is higher. For example, when there are 2 story houses built in like 1930, they were sold usually at higher prices than green (1.5Fin) and yellow (1Story) points. But from the box plot we already see that 1.5Fin is less expensive than 1Story. This may be explained by YearBuilt since majority of green 1.5Fin were built between 1900 and 1940 while yellow 1Story between 1940 and 1980 and many were built after 2000.

# Old vs New, One vs Two Story:
# <img src="http://pic122.huitu.com/pic/20190606/1767502_20190606144007071070_0.jpg" width="400px">
# <img src="http://img.sjiyou.com/UploadFiles/image/20170428/20170428110845_1718.jpg" width="400px">
# 

# BTW, do not tell me that you prefer the first (well I know that it is not bad...) <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS_LRlC0dIpSAOdEsKL6BsPkQNwc62PSnpxH1FCswdr4eqbPYB9" width="200px">

# ## Data Preparation

# ### ** Impute Missing Values **

# #### 1. Drop columns where there is a large percentage of missing data

# In[ ]:


print(train.isnull().sum())


# How to define "a large percentage"? Let us try 15% first.

# In[ ]:


# training data
# calculate percentage of missing values
train_missing = pd.DataFrame(train.isnull().sum()/len(train.index) * 100)
train_missing.columns = ['percent']

# flag columns whose missing percentage are larger than 15%
train_missing.loc[train_missing['percent'] > 15, 'column_select'] = True
train_col_select = train_missing.index[train_missing['column_select'] == True].tolist()
train_col_select


# In[ ]:


# test data
test_missing = pd.DataFrame(test.isnull().sum()/len(test.index) * 100)
test_missing.columns = ['percent']
test_missing.loc[test_missing['percent'] > 15, 'column_select'] = True
test_col_select = test_missing.index[test_missing['column_select'] == True].tolist()
test_col_select


# We know that Alley, FireplaceQu, PoolQC, Fence, MiscFeature are all categorical variables and their missing values occupy over 50% of total. They should be dropped. But LogFrontage has 1201 rows which I personally decide to keep.

# In[ ]:


# drop LotFrontage
train_col_select.pop(0)
test_col_select.pop(0)


# In[ ]:


train.drop(train_col_select, inplace = True, axis = 1, errors = 'ignore')
test.drop(test_col_select, inplace = True, axis = 1, errors = 'ignore')

train.head()


# Note that we orginally have 81 variables for training set and 80 for test set.

# In[ ]:


train.shape


# In[ ]:


test.head()


# In[ ]:


test.shape


# #### 2. Use medians for numeric variables and the most frequent values for non-numeric variables to replace NA

# TansformerMixin can be used to define a custom transformer dealing with heterogeneous data types, which basically contains two major parts:
# - fit: Uses the input data to train the transformer
# - transform: Takes the input features and transforms them

# In[ ]:


from sklearn.base import TransformerMixin
class MissingDataImputer(TransformerMixin):
    def fit(self, X, y=None):
        """Extract mode for categorical features and median for numeric features"""        
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self

    def transform(self, X, y=None):
        """Replace missingness with the array got from fit"""
        return X.fillna(self.fill)


# In[ ]:


train_nmissing = MissingDataImputer().fit_transform(train.iloc[:,1:-1])
test_nmissing = MissingDataImputer().fit_transform(test.iloc[:,1:])
train_nmissing.head()


# In[ ]:


print(train_nmissing.isnull().sum())


# ### **Transform the Data**

# #### 1. For categorical variables, apply onehot encoding

# 1)  Check if training data and test data have the same categorical variables

# In[ ]:


train_cat = train_nmissing.select_dtypes(include=['object']).columns
test_cat = test_nmissing.select_dtypes(include=['object']).columns
train_cat.difference(test_cat)


# 2) Create dummy variables

# I am not using LabelEncoder here because for most categorical variables their values are not in order. I use one hot encoding instead. To ensure that the output matrices are in the same order as in training and test data, I decide to create the one-hot arrays with get_dummies instead of sklearn OneHotEncoder.

# In[ ]:


train_w_dummy = pd.get_dummies(train_nmissing, prefix_sep='_', drop_first=True, columns=train_cat)
test_w_dummy = pd.get_dummies(test_nmissing, prefix_sep='_', drop_first=True, columns=test_cat)

# find all dummy variables in the training set
cat_dummies = [col for col in train_w_dummy 
               if '_' in col 
               and col.split('_')[0] in train_cat]


# 3)  Remove additional variales and add missing variables in test data

# In[ ]:


# drop dummy variables in test set but not in training set
for col in test_w_dummy.columns:
    if ("_" in col) and (col.split("_")[0] in train_cat) and col not in cat_dummies:
        test_w_dummy.drop(col, axis=1, inplace=True)

# add dummy variables in training set but not in test set, and assign them 0
for col in cat_dummies:
    if col not in test_w_dummy.columns:
        test_w_dummy[col] = 0        


# 4) Make sure that variables in test data have the same order as in training data

# In[ ]:


train_cols = list(train_w_dummy.columns[:])
test_w_dummy = test_w_dummy[train_cols]


# Remember that we have 76 variables for training and 75 for test before...

# In[ ]:


train_w_dummy.shape


# In[ ]:


test_w_dummy.shape


# #### 2. For numeric variables, normalize the data

# 1) Check if training data and test data have the same numeric variables

# In[ ]:


train_num = train_nmissing.select_dtypes(exclude=['object']).columns
test_num = test_nmissing.select_dtypes(exclude=['object']).columns
test_num.difference(train_num)


# 2) Normalize test data: standardization or Yeo-Johnson transformation

# As we have experimented before, standardization does not change the shape of the distribution in nature but log transformation does. Log transformation requires data to be positive while Yeo-Johnson transformation supports both positive or negative data. 

# I simply choose skewness as the criterion to select between standardization and Yeo-Johnson transformation because I am not training my data on linear regression models. There are of course more rigorous criteria.

# In[ ]:


train_num_std = [col for col in train_num if abs(train_w_dummy[col].skew()) <= 1]
train_num_yjt = [col for col in train_num if abs(train_w_dummy[col].skew()) > 1]


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer

# standardization
scaler = StandardScaler().fit(train_w_dummy[train_num_std].values)
train_w_dummy[train_num_std] = scaler.transform(train_w_dummy[train_num_std].values)
test_w_dummy[train_num_std] = scaler.transform(test_w_dummy[train_num_std].values)

# power transform
pt = PowerTransformer().fit(train_w_dummy[train_num_yjt].values)
train_w_dummy[train_num_yjt] = pt.transform(train_w_dummy[train_num_yjt].values)
test_w_dummy[train_num_yjt] = pt.transform(test_w_dummy[train_num_yjt].values)


# In[ ]:


test_w_dummy.head()


# 3. Apply Principal Component Analysis

# To address the multicolliearity problem, I apply PCA to decrease the number of variables.

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA().fit(train_w_dummy)
plt.figure(figsize = (6, 4))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components', fontsize = 14)
plt.ylabel('cumulative explained variance', fontsize = 14)

plt.grid(True);


# It is clear to notice that the first 100 components contain nearly 100% of the variance.

# In[ ]:


pca = PCA(n_components = 100)
x_train = pca.fit_transform(train_w_dummy) 
x_test = pca.transform(test_w_dummy) 


# Identify the most important features with names on the first 25 principle components.

# In[ ]:


feature_names = []
for i in range(100):
    # get the index of the feature with the largest absolute value
    feature_idx = np.abs(pca.components_[i]).argmax()
    feature_names.append(train_w_dummy.columns[feature_idx])
    
feature_dict = {'PC{}'.format(i+1): feature_names[i] for i in range(100)}
pd.DataFrame(list(feature_dict.items()), columns=['PC', 'Name']).head(25)


# ## **Start Training**

# I am not going to use regression models for now because of the assumptions they make on the data. And my transformation is just to speed up convergence during training. Instead, I will try with random forests, boosting, and neural networks.

# In[ ]:


# do not forget to log transform our response variable
y_train = train['SalePrice'].values
y_train_log = np.log1p(train['SalePrice']).values

y_test_data = pd.read_csv('../input/sample_submission.csv')
y_test = y_test_data['SalePrice'].values
y_test_log = np.log1p(y_test_data['SalePrice']).values


# #### 1. Random Forest
# 

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from pprint import pprint
rf_base = RandomForestRegressor(n_estimators=400)

# look at parameters used by our base forest
pprint(rf_base.get_params())


# I will start from the original y_train to train the random forest and then use the log transformed y_train to train.

# In[ ]:


# base model result
from sklearn import metrics

# model with original y_train
rf_base.fit(x_train, y_train)
y_pred_rf_base = rf_base.predict(x_test)
# create a dictionary to store mse for comparison, since the results are not stable
mse_dict = {'rf_base': metrics.mean_squared_error(y_test, y_pred_rf_base)}
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_rf_base))  
print('Mean Squared Error:', mse_dict['rf_base'])  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf_base)))


# In[ ]:


rf_base.score(x_train, y_train)


# In[ ]:


# model with log transformed y_train
rf_base.fit(x_train, y_train_log)
# need to take exponential before calculating mse, etc. in order to compare
y_pred_rf_base_log = np.exp(rf_base.predict(x_test))
mse_dict.update({'rf_base_log': metrics.mean_squared_error(y_test, y_pred_rf_base_log)})
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_rf_base_log))  
print('Mean Squared Error:', mse_dict['rf_base_log'])  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf_base_log)))


# In[ ]:


rf_base.score(x_train, y_train_log)


# In[ ]:


y_pred_rf_base_log


# Comparing the metrics, Log transformation is helpful in the random forest case. Both of the R squared are high though. So we will go with the log transformed y_train.

# **Hyperparameter tuning**

# 1) Randomized search

# Compared to grid search, randomized search is less time-consuming so we start from a wider range of parameters with randomized search. In this case, 100 sets of parameters are sampled from all combinations to train.

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

# params that will be sampled from
max_depth = [int(x) for x in np.linspace(40, 80, num = 5)]
max_depth.append(None)
random_params = {'n_estimators': [200, 400, 600, 800, 1000, 1200],
                'max_depth': max_depth,
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2, 4]}

rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_params, n_iter = 100, cv = 3, n_jobs = -1, verbose = 2, random_state = 1)
rf_random.fit(x_train, y_train_log)
rf_random.best_params_


# In[ ]:


rf_random.best_score_


# In[ ]:


# random search with best performance parameters
rf_random_best = rf_random.best_estimator_
y_pred_rf_random = np.exp(rf_random_best.predict(x_test))
mse_dict.update({'rf_random': metrics.mean_squared_error(y_test, y_pred_rf_random)})
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_rf_random))  
print('Mean Squared Error:', mse_dict['rf_random'])  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf_random)))


# In[ ]:


y_pred_rf_random


# The random search model does not yield a better model (sometimes it does not). Judging from the errors, the differences are quite small. But remember these are log transformed results. To save time, I do not include more parameter settings but it definity worths a try.

# 2) Grid search

# Basically, grid search considers all the combinations of parameters so it takes longer time than randomized search. I would use more refined tuning through grid search.

# In[ ]:


from sklearn.model_selection import GridSearchCV
grid_params = {'max_depth': [45, 50, 55, None],
               'min_samples_leaf': [1, 2],
               'min_samples_split': [2, 4, 5],
               'n_estimators': [800, 900, 1000]}

rf_grid = GridSearchCV(estimator = rf, param_grid = grid_params, cv = 3, n_jobs = -1, verbose = 2)
rf_grid.fit(x_train, y_train_log)
rf_grid.best_estimator_


# In[ ]:


rf_grid.best_score_


# In[ ]:


# grid search with best performance parameters
rf_grid_best = rf_grid.best_estimator_
y_pred_rf_grd = np.exp(rf_grid_best.predict(x_test))
mse_dict.update({'rf_grd': metrics.mean_squared_error(y_test, y_pred_rf_grd)})
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_rf_grd))  
print('Mean Squared Error:', mse_dict['rf_grd'])  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf_grd)))


# Compared to random search model, grid search model performs a little better (sometimes it could be the reverse).

# In[ ]:


# keep the output with the lowest mse
rf_model = min(mse_dict, key = lambda x: mse_dict.get(x))
prediction = pd.DataFrame(globals()["y_pred_" + rf_model], columns = ['SalePrice'])
result = pd.concat([y_test_data['Id'], prediction], axis = 1)
result.to_csv('./submission.csv', index = False)


# #### 2. XGBoost

# In[ ]:


from xgboost import XGBRegressor
xgb_base = XGBRegressor()

# current parameters used by XGBoost
pprint(xgb_base.get_params())


# Same with random forest, I will compare the results between model with original y_train and model with log transformed y_train.

# In[ ]:


# model with original y_train
xgb_base.fit(x_train, y_train)
y_pred_xgb_base = xgb_base.predict(x_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_xgb_base))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_xgb_base))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_xgb_base)))


# In[ ]:


xgb_base.score(x_train, y_train)


# In[ ]:


# model with log transformed y_train
xgb_base.fit(x_train, y_train_log)
y_pred_xgb_base_log = np.exp(xgb_base.predict(x_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_xgb_base_log))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_xgb_base_log))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_xgb_base_log)))


# In[ ]:


xgb_base.score(x_train, y_train_log)


# Noticeably, MSE and RMSE of model with y_train_log are much lower than those of model with original y_train; however, R^2 score of model with y_train_log is also a little lower. 

# R^2 score is useful when we are trying to use independent variables to explain the variances in the dependent variable. But here we care MSE more since we are trying to capture the values of house prices. So in the XGBoost case, log tansformation works as well.

# 1) Randomized Search

# In[ ]:


random_params = {'learning_rate': [0.01],
                 'n_estimators': [400, 800, 1000, 1200],
                 'max_depth': [3, 5, 8],
                 'min_child_weight': [4, 6, 8],
                 'subsample': [0.8],
                 'colsample_bytree': [0.8],
                 'reg_alpha': [0, 0.005, 0.01],
                 'seed': [12]}

xgb_random = RandomizedSearchCV(estimator = xgb_base, param_distributions = random_params, n_iter = 100, cv = 3, n_jobs = -1, verbose = 2, random_state = 12)
xgb_random.fit(x_train, y_train_log)
xgb_random.best_params_


# In[ ]:


xgb_random.best_score_


# In[ ]:


xgb_random_best = xgb_random.best_estimator_
y_pred_xgb_random = np.exp(xgb_random_best.predict(x_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_xgb_random))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_xgb_random))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_xgb_random)))


# 2) Grid Search

# In[ ]:


grid_params = {'learning_rate': [0.01],
               'n_estimators': [400, 800, 1000, 1200],
               'max_depth': [3, 5, 8],
               'min_child_weight': [4, 6, 8],
               'subsample': [0.8],
               'colsample_bytree': [0.8],
               'reg_alpha': [0, 0.005, 0.01],
               'seed': [12]}

xgb_grid = GridSearchCV(estimator = xgb_base, param_grid = grid_params, cv = 3, n_jobs = -1, verbose = 2)
xgb_grid.fit(x_train, y_train_log)
xgb_grid.best_estimator_


# In[ ]:


xgb_grid.best_score_


# In[ ]:


# grid search with best performance parameters
xgb_grid_best = xgb_grid.best_estimator_
y_pred_xgb_grd = np.exp(xgb_grid_best.predict(x_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_xgb_grd))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_xgb_grd))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_xgb_grd)))


# #### 3. Neural Networks

# I build a neural network model in TensorFlow with two densely connected hidden layers, and an output layer that returns a single, continuous value. The model building steps are wrapped in a function, neural_net_model.

# In[ ]:


import tensorflow as tf

input_dim = x_train.shape[1]
learning_rate = 0.002
n_nodes_l1 = 25
n_nodes_l2 = 25

x = tf.placeholder("float")
y = tf.placeholder("float")

def neural_net_model(data, input_dim):
    # 2 hidden layer feed forward neural net
    layer_1 = {'weights':tf.Variable(tf.random_normal([input_dim, n_nodes_l1])),
               'biases':tf.Variable(tf.random_normal([n_nodes_l1]))}

    layer_2 = {'weights':tf.Variable(tf.random_normal([n_nodes_l1, n_nodes_l2])),
               'biases':tf.Variable(tf.random_normal([n_nodes_l2]))}
    
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_l2, 1])),
                    'biases':tf.Variable(tf.random_normal([1]))}
    # affine function
    l1 = tf.add(tf.matmul(tf.cast(data, tf.float32), layer_1['weights']), layer_1['biases'])
    # relu activation
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, layer_2['weights']), layer_2['biases'])
    l2 = tf.nn.relu(l2)
    
    output = tf.add(tf.matmul(l2, output_layer['weights']), output_layer['biases'])

    return output


# As before, I will also expriment with original y_train and log transformed y_train_log in the neural network model as well.

# In[ ]:


# train neural network with y_train
# get predictions, define loss and optimizer
prediction = neural_net_model(x_train, input_dim)
cost = tf.reduce_mean(tf.square(prediction - y_train))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

batch_size = 100
epochs = 500
display_epoch = 100
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Training cycle
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(x_train.shape[0]/batch_size)
        for i in range(total_batch-1):
            batch_x = x_train[i*batch_size:(i+1)*batch_size]
            batch_y = y_train[i*batch_size:(i+1)*batch_size]
            
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})               
            avg_cost += c/total_batch
            
        if epoch % display_epoch == 0:    
            print("Epoch:", (epoch + 1), " mse =", "{:.6f}".format(avg_cost)) 
    
    # running test set
    results = sess.run(prediction, feed_dict={x: x_test})
    test_cost = sess.run(cost, feed_dict={x: x_test, y: y_test})
    print('test cost: {:.6f}'.format(test_cost))
    
    # calculate r^2
    total_error = tf.reduce_sum(tf.square(y_test - tf.reduce_mean(y_test)))
    unexplained_error = tf.reduce_sum(tf.square(y_test - results))
    R_squared = 1.0 - tf.div(total_error, unexplained_error)
    print(R_squared.eval())    


# In[ ]:


# train neural network with y_train_log
# get predictions, define loss and optimizer
prediction = neural_net_model(x_train, input_dim)
cost = tf.reduce_mean(tf.square(prediction - y_train_log))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

batch_size = 100
epochs = 500
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Training cycle
    for epoch in range(epochs):
        avg_cost = 0        
        total_batch = int(x_train.shape[0]/batch_size)
        for i in range(total_batch-1):
            batch_x = x_train[i*batch_size:(i+1)*batch_size]
            batch_y = y_train_log[i*batch_size:(i+1)*batch_size]
            
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})               
            avg_cost += c/total_batch

        if epoch % display_epoch == 0:    
            print("Epoch:", (epoch + 1), " mse =", "{:.6f}".format(avg_cost))
    
    # running test set
    results = sess.run(prediction, feed_dict={x: x_test})
    test_cost = tf.reduce_mean(tf.square(tf.math.exp(results) - y_test))
    print(test_cost.eval())
    
    total_error = tf.reduce_sum(tf.square(y_test_log - tf.reduce_mean(y_test_log)))
    unexplained_error = tf.reduce_sum(tf.square(y_test_log - results))
    R_squared = 1.0 - tf.div(total_error, unexplained_error)
    print(R_squared.eval())


# According to the above outputs, the neural network models seem to perform not as well as random forests and xgboost. The R squared is almost 100% but our key metric is mean squared error. However, the high R Squared from these models indicates that the 100 selected components have great explanatory power for the house prices. But for the submission, I will still use the best result based on random forests. 

# Finally, wish everyone has a great house to call home~ And, do not forget to upvote my kernel if you find it helpful, thanks~

# <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTpyyL7_QzWk7T9Qcri6me8NK1R6Qcu6LdzWUyCirlJRG1sZyga0w" width="300px">
