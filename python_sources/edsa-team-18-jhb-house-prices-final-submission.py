#!/usr/bin/env python
# coding: utf-8

# # Overview

# **This notebook presents the steps followed in creating our team's submission for the House Prices: Advanced Regression Techniques competiton, which we completed as part of an assignment**
# 
# ![](https://cdn.pixabay.com/photo/2015/10/26/21/11/houses-1007932_960_720.jpg)
# *Have you ever wondered what determines a house's sale price? Join us as we attempt to find out and build a predictive model for house prices based on the Ames dataset!*
# 
# **Sections contained within the notebook are as follows:**
# 
# **1) Exploratory Data Analysis**
# * **Importing the data**
# * **The target variable: SalePrice**
# * **Relationship of numerical features to SalePrice**
# * **Relationship of categorical features to SalePrice**
# 
# **2) Data Cleaning**
# * **Removal of outliers**
# * **Dealing with null values**
# * **Preparing data for model fitting**
# 
# **3) Regression Models and Prediction of SalePrice**
# * **Linear, Ridge, Lasso and RandomForest**

# ## Library Imports 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# # 1) Exploratory Data Analysis

# ## Importing the data

# In[2]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[3]:


train_df.head()


# In[4]:


test_df.head()


# In[5]:


print(train_df.info())
print(test_df.info())


# **From this first look at the dataset, we can see the following:**
# * The dependent variable we are trying to predict is SalePrice (absent from test set)
# * There are 79 features (independent variables) given to use for building the model, plus Id columns for each dataset
# * Number of **categorical** features = **43**
# * Number of **numerical** features = **36**
# * Full descriptions of each feature are given in the text file provided for the competition
# 
# **We store the Ids of the test dataset for later in order to compile the submission file:**

# In[6]:


ids_test = test_df['Id']


# ## The target variable: SalePrice

# **As a first step, we look at the summary statistics**

# In[7]:


train_df['SalePrice'].describe()


# **Next, we assess the distribution of SalePrice to check normality**
# 
# From the distribution plot below, it is clear that the variable is skewed to the right. We will correct the skewness by log transforming the variable

# In[8]:


f, ax = plt.subplots(figsize=(12, 9))
sns.distplot(train_df['SalePrice']).set_title('Distribution of SalePrice')


# In[9]:


train_df['SalePrice'] = np.log1p(train_df['SalePrice'])


# In[10]:


f, ax = plt.subplots(figsize=(12, 9))
sns.distplot(train_df['SalePrice']).set_title('Distribution of log[SalePrice]')


# The distribution plot after log transformation shows that the target variable is now approximately normally distributed

# ## Relationship of numerical features to SalePrice

# **To get a broad overview of relationships between variables, a heatmap of the correlations is plotted**
# 
# The colour of each square represents the correlation coefficient between a variable on the vertical axis and its corresponding variable on the horizontal axis. This enables us to visually identify high and low correlations

# In[11]:


corrmatrix = train_df.corr() #Create correlation matrix
f, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(corrmatrix, vmax=.8, square=True, cmap='BuPu')


# We can also look at the values of the correlation coefficients

# In[12]:


pd.DataFrame(corrmatrix['SalePrice'].abs().sort_values(ascending=False))


# Some variables highly correlated with SalePrice are 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea' and 'TotalBsmtSF'. Many of these make intuitive sense. For example, we would expect houses with higher overall quality and size to sell for higher. For now, we take note of variables that show very low correlations, but will only decide which to remove from the dataset after further analysis.
# 
# **Further investigating the relationship between SalePrice and continuous numerical variables can be done using scatter plots**
# 
# Note: Only selected plots are shown below to illustrate the point (3 variables with high correlation to SalePrice and 1 with low correlation)

# In[13]:


#GrLivArea - high correlation
with sns.axes_style('white'):
    sns.jointplot(x=train_df['GrLivArea'],y=train_df['SalePrice'], color='firebrick')


# In[14]:


#GarageArea - high correlation
with sns.axes_style('white'):
    sns.jointplot(y=train_df['SalePrice'], x=train_df['GarageArea'], color='firebrick')


# In[15]:


#TotalBsmtSF - high correlation
with sns.axes_style('white'):
    sns.jointplot(y=train_df['SalePrice'], x=train_df['TotalBsmtSF'], color='firebrick')


# In[16]:


#PoolArea - low correlation
with sns.axes_style('white'):
    sns.jointplot(x=train_df['SalePrice'], y=train_df['PoolArea'], color='teal')


# **Correlations between independent variables should also be investigated to ensure that redundant variables are removed**

# In[17]:


highcorr = pd.DataFrame(corrmatrix.abs().unstack().transpose().sort_values(ascending=False).drop_duplicates())
highcorr.head(15)


# ## Relationship of categorical variables to SalePrice
# 
# **Similarly to scatter plots for numerical variables, box plots provide a way to visualize the relationship between the categorical variables and SalePrice** 
# 
# Note: Only selected plots are shown below to illustrate the point (3 variables with high correlation to SalePrice and 1 with low correlation)

# In[18]:


#Neighborhood - high correlation
ax=sns.catplot(x='Neighborhood', y='SalePrice', kind='boxen',data=train_df.sort_values('Neighborhood'),height=12,aspect=2)
ax.set_xticklabels(size=15,rotation=30)
ax.set_yticklabels(size=15,rotation=30)
plt.xlabel('Neighborhood',size=25)
plt.ylabel('SalePrice',size=25)
plt.show()


# In[19]:


#ExterQual - high correlation
ax=sns.catplot(x='ExterQual', y='SalePrice',kind='boxen',data=train_df.sort_values('BldgType'),height=12,aspect=2)
ax.set_xticklabels(size=15,rotation=30)
ax.set_yticklabels(size=15,rotation=30)
plt.xlabel('ExterQual',size=25)
plt.ylabel('SalePrice',size=25)
plt.show()


# In[20]:


#BsmtQual - high correlation
ax=sns.catplot(x='BsmtQual', y='SalePrice', kind='boxen',data=train_df.sort_values('BsmtQual'),height=12,aspect=2)
ax.set_xticklabels(size=15,rotation=30)
ax.set_yticklabels(size=15,rotation=30)
plt.xlabel('BsmtQual',size=25)
plt.ylabel('SalePrice',size=25)


# In[21]:


#Heating - low correlation
ax=sns.catplot(x='Heating', y='SalePrice', kind='boxen',data=train_df.sort_values('Heating'),height=12,aspect=2)
ax.set_xticklabels(size=15,rotation=30)
ax.set_yticklabels(size=15,rotation=30)
plt.xlabel('Heating',size=25)
plt.ylabel('SalePrice',size=25)


# # 2) Data Cleaning

# ## Removal of outliers

# Although there are no definitive rules for going about the removal of outliers, they can lead to overfitting
# 
# 
# A conservative approach was taken to remove outliers from only the target variable, as removal of too much data during initial model tests showed poorer performance. We decided to remove entries based on their position when plotting SalePrice against its most correlated numerical variables (see charts above). In this case we removed entries with 'GrLivArea' greater than 4000 square feet. This decision was backed up by [information about the dataset](https://ww2.amstat.org/publications/jse/v19n3/decock.pdf) from the authors, who recommend the removal of these points.

# In[22]:


#GrLivArea vs SalePrice - before
with sns.axes_style('white'):
    sns.jointplot(x=train_df['GrLivArea'],y=train_df['SalePrice'], color='firebrick')


# In[23]:


train_df = train_df[train_df['GrLivArea'] < 4000]


# In[24]:


#GrLivArea vs SalePrice - after
with sns.axes_style('white'):
    sns.jointplot(x=train_df['GrLivArea'],y=train_df['SalePrice'], color='firebrick')


# **For further data manipulation, the test and train datasets are combined to ensure uniform changes to both**

# In[25]:


df = pd.concat([train_df, test_df])


# ## Dealing with null values
# 
# **The percentage of null values for each feature is calculated below**

# In[26]:


percent_null = pd.DataFrame((df.isnull().sum()/df.isnull().count()).sort_values(ascending=False))
percent_null.head(20)


# **Based on this, the following will be performed:**
# 
# * Features with a high percentage of nulls will be removed 
# * Remaining nulls will be replaced, based on feature characteristics
# 
# **We decided to remove columns only if they had 2 or more of the following undesirable characteristics:**
# 1. Low correlation to SalePrice
# 2. High correlation with another variable in the dataset
# 3. High percentage of null values
# 

# In[27]:


to_drop = ['Id', 'PoolQC', 'MiscVal', 'MiscFeature', 'Alley', 'LandContour', 'Utilities', 'FireplaceQu', 'GarageCond', 'Fence']
df.drop(to_drop, axis=1, inplace=True)


# **Remaining null values were imputed as follows:**
# 
# **Categorical Features:**
# 
# * If a null indicates that the feature is absent, it is replaced by 'None'
# * If a null does not indicate such, it is replaced by the most common value (mode)
# 
# **Numerical Features:**
# 
# * If a null indicates that a feature is absent, it is replaced by 0
# * If a null does not indicate such, it is replaced by the median (chosen as opposed to the mean as it is not sensitive to outliers)

# In[28]:


#Categorical features where null indicates the feature is NOT present - will be replaced with the str 'None'
cat_none = ['MasVnrType', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'GarageType', 'GarageFinish']
for col in cat_none:
    df[col].fillna('None', inplace=True)
    
#Categorical features which must be replaced by mode (nulls don't indicate feature is absent)
cat_mode = ['MSZoning', 'Exterior1st', 'Exterior2nd', 'Electrical', 'KitchenQual', 'SaleType', 'Functional', 'GarageQual']
for col in cat_mode:
    df[col].fillna(df[col].mode()[0], inplace=True)
    
#Numerical features where null indicates feature is not present - will be replaced by 0
num_zero = ['MasVnrArea', 'BsmtHalfBath', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'TotalBsmtSF', 'BsmtUnfSF', 'GarageYrBlt', 'GarageArea', 'GarageCars']
for col in num_zero:
    df[col].fillna(0, inplace=True)
    
#Continuous numerical feature - nulls to be replaced by median (Grouped by neighborhood as the LotFrontage is expected to be similar in a particular neighborhood)
df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
    


# ## Preparing data for model fitting
# 
# **The data needs to be in the proper format for fitting regression models. The following remains to be done:**
# * Correcting feature skewness
# * Converting discrete numerical features to categorical
# * Encoding categorical variables

# In[29]:


#Convert discrete numerical features into categorical
numerical_to_cat=['BedroomAbvGr', 'Fireplaces', 'FullBath', 'GarageCars', 'HalfBath', 'KitchenAbvGr', 'OverallQual']

df[numerical_to_cat] = df[numerical_to_cat].apply(lambda x: x.astype("str"))


# In[30]:


#Correct feature skewness - only log those features which show a decrease in skewness after the transformation
numeric_features = df.dtypes[df.dtypes != 'object'].index.drop('SalePrice')
skew_b = df[numeric_features].apply(lambda x: skew(x.dropna()))
log = np.log1p(df[numeric_features])
skew_a = log[numeric_features].apply(lambda x: skew(x.dropna()))
skew_diff = (abs(skew_b)-abs(skew_a)).sort_values(ascending=False)
df[skew_diff[skew_diff > 0].index] = np.log1p(df[skew_diff[skew_diff > 0].index])


# In[31]:


#Encode categorical variables
df = pd.get_dummies(df, drop_first=True)


# **Finally, we split the data back into train and test sets**

# In[32]:


X_train = df.iloc[:1456].drop('SalePrice', axis=1)
X_test = df.iloc[1456:].drop('SalePrice', axis=1)
y_train = df['SalePrice'].dropna().values


# **...And do one last check to make sure our datasets are in order**

# In[33]:


print(X_train.info())
print(X_test.info())


# # 3) Regression Models and Prediction of SalePrice
# 
# **Four regression models were evaluated:**
# * Linear regression
# * Ridge
# * Lasso
# * RandomForest
# 
# **Hyperparameter optimization was performed using GridSearchCV** (note in some cases the value ranges showed here for each model are those after a few rounds of optimization)
# 
# **Performance of models was evaluated using the root mean square error (RMSE) for the training set,** as this is comparable to the scoring metric used for the competition (root mean square log error)

# In[34]:


#Define function to extract best train RMSE from GridSearchCV
def best_rmse(grid):
    
    best_score = np.sqrt(-grid.best_score_)
    print(best_score)    
    print(grid.best_estimator_)
    
    return best_score


# In[35]:


#Linear Regression

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
parameters_lm = {'fit_intercept':[True,False]}
grid_lm = GridSearchCV(lm, parameters_lm, cv=5, verbose=1 , scoring ='neg_mean_squared_error')
grid_lm.fit(X_train, y_train)
score_lm = best_rmse(grid_lm)


# In[36]:


#Ridge

from sklearn.linear_model import Ridge

ridge = Ridge()
parameters_ridge = {'alpha': [4, 4.1, 4.2, 4.3, 4.4, 4.5], 'tol': [0.001, 0.01, 0.1]}
grid_ridge = GridSearchCV(ridge, parameters_ridge, cv=5, verbose=1, scoring='neg_mean_squared_error')
grid_ridge.fit(X_train, y_train)
score_ridge = best_rmse(grid_ridge)


# In[37]:


#Lasso

from sklearn.linear_model import Lasso

lasso = Lasso()
parameters_lasso = {'alpha': [1e-4, 0.001, 0.01, 0.1, 0.5, 1], 'tol':[1e-06, 1e-05, 1e-04, 1e-03, 0.01, 0.1]}
grid_lasso = GridSearchCV(lasso, parameters_lasso, cv=5, verbose=1, scoring='neg_mean_squared_error')
grid_lasso.fit(X_train, y_train)
score_lasso = best_rmse(grid_lasso)


# In[38]:


#RandomForest

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
parameters_rf = {'min_samples_split' : [2, 3, 4], 'n_estimators' : [50, 100, 500]}
grid_rf = GridSearchCV(rf, parameters_rf, cv=5, verbose=1, scoring='neg_mean_squared_error')
grid_rf.fit(X_train, y_train)
score_rf = best_rmse(grid_rf)


# **Comparison of RMSE of the 4 models**

# In[41]:


rmse_x = ['linear', 'ridge', 'lasso', 'randomforest']
rmse_y = [score_lm, score_ridge, score_lasso, score_rf]
sns.set(style='whitegrid')
sns.barplot(x=rmse_x, y=rmse_y)


# **Based on the train RMSE of the models, the top performing two (lasso and ridge) were used to make predictions for the test data and the average of these used for the final submission**

# In[43]:


pred_lasso = np.expm1(grid_lasso.predict(X_test))
pred_ridge = np.expm1(grid_ridge.predict(X_test))

pred_combined = (pred_lasso + pred_ridge)/2


# In[44]:


#Creating dataframe of submission data
submission = pd.DataFrame()
submission['Id'] = ids_test.values
submission['SalePrice'] = pred_combined


# In[45]:


#Save the output as a csv
submission.to_csv('submission_final_edsateam18.csv', index=False)


# In[ ]:




