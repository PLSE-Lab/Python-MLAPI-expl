#!/usr/bin/env python
# coding: utf-8

# ## Objective:
# 
# - The Objective is to predict Sale Price of each house given 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa.
# 
# ## The approach:
# 
# 1. Exploratory Data Analysis to get a sense of the data.
# 
# 2. Feture engineering: 
#     2.1 We will be creating a few features by the combining the existing features. 
#     2.2 We will also fill in the missing values.
#     All of this will be done using sklearn pipelines.
#     
# 3. Training and Predicting:
#     
#     3.1 We will perform Grid search cross validation of 5 different algorithms: Ridge Regression, Elastic Net, SVM, XGBoost, GradientBoost. The hyper parameter space for a few algorithms has been thinned down as it the runtime is too high here which is not the case with a local version.
#     
#     3.2 We will combine the 5 algorithms using a voting regressor and use that to train on the entire dataset and predict on the test set.
#     
#     

# In[ ]:


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Reading the data
train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
sample_submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')


# # 1. EDA

# ### 1.1 Let's try and get a sense of our data

# In[ ]:


# Lets take a look at the first few rows of the dataset
train_df.head()


# In[ ]:


# Lets look at the shape
train_df.shape


# - Seems we have 1460 observations and 79 features (if we exlcude ID and Dependent variable i.e. SalePrice)

# ### 1.2 Let's see how our dependant variable looks

# In[ ]:


# Setting the grid style
sns.set_style('darkgrid')
sns.set_color_codes(palette='dark')

# Setting plot area
f, ax = plt.subplots(figsize=(9, 9))

# plotting the distribution plot
sns.distplot(train_df['SalePrice'], color="m", axlabel='SalePrice')
ax.set(title="Histogram for SalePrice")
plt.show()


# - Seems like SalePrice is right skewed and we will have to normalize it.

# ### 1.3 Let's vizualize the correlation matrix

# In[ ]:


# Calc correlation matrix
corr_mat = train_df.corr()

# Set plot size
plt.subplots(figsize=(12,10))

# Plot heatmap
sns.heatmap(corr_mat, 
            square=True, 
            robust=True, 
            cmap='OrRd', # use orange/red colour map
            cbar_kws={'fraction' : 0.01}, # shrink colour bar
            linewidth=1) # space between cells


# #### 1.3.2 Zoomed in Scatterplot
# - Let's vizualize only the variables with high correlation coefficient.

# In[ ]:


# number of variables we want on the heatmap
k = 10 

# Filter in the Top k variables with highest correlation with SalePrice
cols = corr_mat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_df[cols].values.T)

cmap_ch = sns.cubehelix_palette(as_cmap=True, light=.95)
# Creating the heatmap
hm = sns.heatmap(cm,
                 cmap = cmap_ch,
                 cbar = True, 
                 annot = True, # Since we want to know the the correlation coeff as well.
                 square = True, 
                 robust = True,
                 cbar_kws={'fraction' : 0.01}, # shrink colour bar
                 annot_kws={'size': 8}, # setting label size
                 yticklabels=cols.values, # set y labels
                 xticklabels=cols.values,
                 linewidth=1) # Set xlabels
plt.show()


# - Let's do a quick bivariate analysis of a few features and SalePrice

# ### 1.4 Bivariate Analysis

# #### 1.4.1 SalePrice and OverallQual

# In[ ]:


# Creating a boxplot
chart = sns.catplot(data = train_df ,
                    x = 'OverallQual',
                    y='SalePrice',
                    kind='box',
                    height=8,
                    palette='Set2')

# Setting X axis labels
chart.set_xticklabels(fontweight='light',fontsize='large')


# - Now, Overall quality itself seems to be a calculated metric but it definitely has a high correlation with SalePrice. 
# - Also, within the 10 possible values, the lower values (1,2 etc.) have a lesser spread as compared to higher values (9,10 etc.), which have a few outliers as well.

# #### 1.4.2 SalePrice and YearBuilt

# In[ ]:


# Creating a boxplot
chart = sns.catplot(data = train_df ,x = 'YearBuilt',y='SalePrice',
                    kind='box', # we want a boxplot
                    height=5,
                    aspect=4,
                    palette='Set2')

# Setting X axis labels
chart.set_xticklabels(fontweight='light',
                      fontsize='large',
                      rotation=90,
                      horizontalalignment='center')


# - The Years range from 1872 to 2010. Post mid-30s, the general trend for median house price is going up. However, we don't know if the prices are adjusted for inflation. They most probably aren't.

# #### 1.4.3. SalePrice and TotalBsmtSF

# In[ ]:


plt.figure(figsize=(10,8))

# Creating a scatterplot
sns.scatterplot(data = train_df ,
                x = 'TotalBsmtSF',
                y ='SalePrice',
                alpha = 0.65,
                color = 'g') 


# - The positive correlation is clear.
# - Also, some outliers exist. Vis-a-vis: Point on the right with more than 6000 sq ft basement and a low SalePrice. (I wonder if this is a bunker!)

# #### 1.4.4. SalePrice and GrLivArea

# In[ ]:


plt.figure(figsize=(10,8))

# Creating a scatterplot
sns.scatterplot(data = train_df ,
                x = 'GrLivArea',
                y ='SalePrice',
                alpha = 0.65,
                color = 'b') 


# - Overall, the positive corrrelation is quite obvious.
# - There are 2 sets of outliers here:
#     - The 2 observations on the bottom right, with >4000 sqft of GrLivArea but < 200k of SalePrice. We could consider drropping these points.
#     - The 2 observations on the top right, with >4000 sqft of GrLivArea and >700k of SalePrice. Given, that these observations, although extreme, are still on the same trend as the other observations, we should keep them.

# #### 1.4.5. SalePrice and GarageCars

# In[ ]:


# Creating a boxplot
chart = sns.catplot(data = train_df ,
                    x = 'GarageCars',
                    y='SalePrice',
                    kind='box',
                    height=6,
                    palette='Set2')

# Setting X axis labels
chart.set_xticklabels(fontweight='light',fontsize='large')


# - The definition for GarageCars is "Size of garage in car capacity". You would expect SalePrice to keep on increasing with the garage size. However, median SalePrice decreases for the highest car capacity bucket. It could be because of a smaller sample size.
# - Further, looking at the heatmap, the variables 'GarageArea' and 'GarageCars' seem to be highly correlated. This could lead to multicollinearity. 
# - Also, GarageCars seems to be correlated to OverallQual, indicating that it could be one of the factors involved in calculating OverallQual (assuming it is a calcualted variable).

# # 2. Feature engineering

# #### 2.1 Removing IDs
# - We need to remove the ID column since it is a unique identifier

# In[ ]:


# Storing the IDs in a separate DF
train_df_IDs = train_df['Id']
test_df_IDs = test_df['Id']

# Dropping the columns
train_df.drop(['Id'], axis=1, inplace=True)
test_df.drop(['Id'], axis=1, inplace=True)

# Checking the shape of both DFs
print(train_df.shape) 
print(test_df.shape)


# - The numbers of columns have reduced by 1. ID has been removed.

# #### 2.2 Normalizing SalePrice
# 
# - As we saw earlier, SalePrice is not normally distributed. ML models don't perform well with skewed data.
# - In order to fix this, we will do a log transformation of SalePrice

# In[ ]:


# Log transforming SalePrice
train_df["SalePrice_log"] = np.log(train_df["SalePrice"])

# Plotting to vizualize the transformed variable
sns.distplot(train_df['SalePrice_log'], color="m", axlabel='SalePrice_log')


# - SalePrice looks normally distributed now.

# In[ ]:


# Dropping SalePrice_log as we will clean it in the next steps with pipeline
train_df = train_df.drop('SalePrice_log',axis=1)


# #### 2.3 Removing Outliers
# 
# - We looked at the 2 observation on the 'GrLivArea - SalePrice' scatterplot. Let us remove them.

# In[ ]:


# Dropping the outliers
train_df.drop(train_df[(train_df['GrLivArea']>4500) & (train_df['SalePrice']<300000)].index, inplace=True)

# Vizualizing the new scatterplot
plt.figure(figsize=(10,8))

# Creating a scatterplot
sns.scatterplot(data = train_df ,
                x = 'GrLivArea',
                y ='SalePrice',
                alpha = 0.65,
                color = 'b') 


# #### 2.4 Missing Data

# In[ ]:


# Separating Predictor and Labels
housing = train_df.drop("SalePrice",axis=1)
housing_labels = train_df['SalePrice']


# In[ ]:


# calc total missing values
total_series = housing.isnull().sum().sort_values(ascending=False)

# calc percentages
perc_series = (housing.isnull().sum()/housing.isnull().count()).sort_values(ascending = False)

# concatenating total values and percentages
missing_data = pd.concat([total_series, perc_series*100], axis=1, keys=['Total #', 'Percent'])

# Looking at top 20 entries
missing_data.head(20)


# - We have at least 19 columns with missing values.
# - Additionally, there are some columns like MSSubClass,YrSold,MoSold,GarageYrBlt that should cateegorical but are listed as numeric

# In[ ]:


# converting numeric columns to categorical
cols_int_to_str = ['MSSubClass','YrSold','MoSold','GarageYrBlt']

for col in cols_int_to_str:
    housing[col] = housing[col].astype(str)
    test_df[col] = test_df[col].astype(str)


# - It is important to take a step back here and go through the data description. Obviously, we have 2 types of data points. Numeric and categorical. Further, for some categorical features, missing values can be filled with "None", whereas for some there is no clear intuition so we will go ahead with the most frequent value. Let's split the dataframe in 3 categories: 
#     - Numeric
#     - Categorical to be filled with "None"
#     - Categorical to be filled with most frequent

# In[ ]:


# Creating a list of numerics we want for the mask
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

# Creating a dataframe for numeric features
housing_num = housing.select_dtypes(include=numerics)
print(housing_num.shape)


# In[ ]:


# List of Categorical features that are to be filled with None
cat_none = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','BsmtQual', 'BsmtCond',            'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
# Creating a dataframe of features for "none"
housing_cat_none = housing[cat_none]

# All the other categorical features
housing_cat_freq = housing[(housing.columns.difference(cat_none)) & (housing.columns.difference(housing_num.columns))]


# #### 2.5 Numerical Pipeline
# 
# - We will be creating 3 addtitional features:
#         - Total_sqr_footage: The total sqr footage of the house i.e. basement + 1st floot+ 2nd floor.
#         - Total_Bathrooms: Total number of bathrooms.
#         - Total_porch_sf: Total sqr footage of the porch.
#         
# This idea for the features is being leveraged from Lavanya Shukla's Kernel (see resoruces section)

# In[ ]:


# Importing the modules
from sklearn.base import BaseEstimator, TransformerMixin

# getting index of relevant columns instead of hardcoding
BsmtFinSF1_ix, BsmtFinSF2_ix, flr_1_ix, flr_2_ix,FullBath_ix, HalfBath_ix, BsmtFullBath_ix, BsmtHalfBath_ix,OpenPorchSF_ix, SsnPorch_ix, EnclosedPorch_ix, ScreenPorch_ix, WoodDeckSF_ix = [
    list(housing_num.columns).index(col)
    for col in ("BsmtFinSF1", "BsmtFinSF2","1stFlrSF","2ndFlrSF",\
                "FullBath","HalfBath","BsmtFullBath","BsmtHalfBath",\
                "OpenPorchSF","3SsnPorch","EnclosedPorch","ScreenPorch","WoodDeckSF")]

# Creating CombinedAttributesAdder class for creating the features
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, Total_sqr_footage = True, Total_Bathrooms=True,Total_porch_sf=True): 
        self.Total_sqr_footage = Total_sqr_footage
        self.Total_Bathrooms = Total_Bathrooms
        self.Total_porch_sf = Total_porch_sf
        
    def fit(self, X, y=None):
        return self 
    
    def transform(self, X, y=None):
        if self.Total_sqr_footage: # Calculate total footage
            Total_sqr_footage = X[:, BsmtFinSF1_ix] + X[:, BsmtFinSF2_ix] + X[:,flr_1_ix] + X[:,flr_2_ix]
       
        if self.Total_Bathrooms: # Calculate total bathrooms
            Total_Bathrooms = X[:, FullBath_ix] + X[:, HalfBath_ix] + X[:,BsmtFullBath_ix] + X[:,BsmtHalfBath_ix]
            
        if self.Total_porch_sf: # Calculate total porch area
            Total_porch_sf = X[:, OpenPorchSF_ix] + X[:, SsnPorch_ix] + X[:,EnclosedPorch_ix] + X[:,ScreenPorch_ix] + X[:,WoodDeckSF_ix]
            
        return np.c_[X, Total_sqr_footage,Total_Bathrooms,Total_porch_sf]
    


# In[ ]:


# Importing necessary libraries
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Creating numerical pipeline
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="mean")), # To impute na values with mean 
        ('attribs_adder', CombinedAttributesAdder()), # Create new features
        ('std_scaler', StandardScaler()), # Scale the numeric features
    ])


# #### 2.6 Categorical Pipeline

# In[ ]:


# Importing necessary libraries
from sklearn.preprocessing import OneHotEncoder

# Creating pipeline for categorical variables with missing value should be "None"
cat_pipeline_none = Pipeline([
        ('imputer', SimpleImputer(strategy='constant',fill_value='None')),
        ('encoder', OneHotEncoder(sparse=False,handle_unknown='ignore'))
    ])

# Creating pipeline for categorical variables where we plug missing value with most frequent value
cat_pipeline_freq = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(sparse=False,handle_unknown='ignore'))
    ])


# #### 2.7 Creating Full pipeline

# In[ ]:


# Importing ColumnTransformer
from sklearn.compose import ColumnTransformer

# Creating full pipeline to process numeric and categorical features
full_pipeline = ColumnTransformer(transformers=[
        ("num", num_pipeline, housing_num.columns),
        ("cat_none", cat_pipeline_none, housing_cat_none.columns),
        ("cat_freq", cat_pipeline_freq, housing_cat_freq.columns),
    ])

# Instatiating the full pipelines object
transf = full_pipeline.fit(housing)

# Creating prepared data by passing the training set without labels
housing_prepared = transf.transform(housing)


# In[ ]:


# Checking the shape of the newly created data
housing_prepared.shape


# In[ ]:


# Cleaning the test data 
test_prepared = transf.transform(test_df)


# In[ ]:


# Checking Test data's shape
test_prepared.shape


# # 3. Training Models

# In[ ]:


# Defining CV Score function we will use to calculate the scores
def cv_score(score):
    rmse = np.sqrt(-score) # -score because we are using "neg_mean_squared_error" as our metric
    return (rmse)


# ### 3.1 RandomForestRegressor
# 
# - In all honesty, I have removed a few values from the parameter grid as the kernel takes too long to run. 
# - The runtime would be far better on a local version of the notebook and one can consider a borader hyperparameter space.

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# Import RandomForestRegressor\nfrom sklearn.ensemble import RandomForestRegressor\nfrom sklearn.model_selection import GridSearchCV\nfrom sklearn.metrics import mean_squared_error as MSE\n\n# Instantiate RandomForestRegressor\nrf = RandomForestRegressor(random_state = 42)\n\n# Creating Parameter grid for GridSearch CV\nparams_rf = {\n    'n_estimators': [500,1000], # No of trees\n    'max_depth': [10,15], # maximum depth to explore\n    'min_samples_split':[5], # minimum samples required for split\n    'min_samples_leaf':[5], # minimum samples required at leaf\n    'max_features': [ 'auto'] # number of features for the best split\n}\n\n# Instantiate grid_rf\ngrid_rf = GridSearchCV(estimator = rf, # regressor we want to use\n                       param_grid = params_rf, # Hyperparameter space\n                       scoring ='neg_mean_squared_error', # MSE will be performance metric\n                       cv = 3, # #of folds\n                       verbose = 1,\n                       n_jobs = -1) # use all cores\n\n# fit the model\ngrid_rf.fit(housing_prepared,housing_labels)")


# In[ ]:


# Lets look at the Cross Validation score for RandomForestRegressor
print('CV Score for best RandomForestRegressor model: {:.2f}'.format(cv_score(grid_rf.best_score_)))


# In[ ]:


# Store the best model 
best_model_RF = grid_rf.best_estimator_


# ### 3.2 Gradient Boosting
# 
# - GradientBoostingRegressor could have a very high runtime (for a small learning rate), so I have removed thinned down the hyperparam space considered here. I tested out a lot more values on my local version. 

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# Importing the necessary package\nfrom sklearn.ensemble import GradientBoostingRegressor\n\n# Instantiate the Gradient Boosting Regressor\ngbr = GradientBoostingRegressor(subsample = 0.9, # this is essentially stochastic gradient boosting\n                                max_features = 0.75,\n                                random_state = 42,\n                                warm_start = True,\n                                learning_rate= 0.01) # low learning rate\n\n# Creating Parameter grid for GridSearch CV\nparams_gbr = {\n    'n_estimators': [8000], # Given that the learning rate is very low, we are increasing the num of estimators\n    'max_depth': [2,3], \n    'min_samples_split':[5],\n    'min_samples_leaf':[5],\n    'max_features': ['sqrt']\n}\n\n# Instantiate grid search using GradientBoostingRegressor\ngrid_gbr = GridSearchCV(estimator = gbr, # regressor we want to use\n                       param_grid = params_gbr, # Hyperparameter space\n                       scoring ='neg_mean_squared_error',\n                       cv = 3, # No of folds\n                       verbose = 1,\n                       n_jobs = -1) # use all cores\n\n# fit the model\ngrid_gbr.fit(housing_prepared,housing_labels)")


# In[ ]:


# Lets look at the Cross Validation score for GradientBoostingRegressor
print('CV Score for best GradientBoostingRegressor model: {:.2f}'.format(cv_score(grid_gbr.best_score_)))


# In[ ]:


# Store the best model 
best_model_GBR = grid_gbr.best_estimator_


# ### 3.3 Ridge Regression

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nfrom sklearn.linear_model import Ridge\n\n# Instantiate the Ridge Regressor\nridge = Ridge(random_state=42)\n\n# Creating Parameter grid for GridSearch CV\nparams_ridge = {\n    'alpha': [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100], #L2 parameter space\n    'solver': ['auto','saga','sag','cholesky']\n}\n\n# Instantiate grid search using GradientBoostingRegressor\ngrid_ridge = GridSearchCV(estimator = ridge, # regressor we want to use\n                       param_grid = params_ridge, # Hyperparameter space\n                       scoring ='neg_mean_squared_error',\n                       cv = 3, # No of folds\n                       verbose = 1,\n                       n_jobs = -1) # use all cores\n\n# fit the model\ngrid_ridge.fit(housing_prepared,housing_labels)")


# In[ ]:


# Lets look at the Cross Validation score for RidgeRegressor
print('CV Score for best RidgeRegressor model: {:.2f}'.format(cv_score(grid_ridge.best_score_)))


# In[ ]:


# Store the best model 
best_model_ridge = grid_ridge.best_estimator_


# ### 3.4 ElasticNet Regression

# In[ ]:


get_ipython().run_cell_magic('time', '', "from sklearn.linear_model import ElasticNet\n\n# Instantiate the Ridge Regressor\nelastic = ElasticNet(random_state=42)\n\n# Creating Parameter grid for GridSearch CV\nparams_elastic = {\n    'alpha': [ 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15], #L2 regularization parameter space\n    'l1_ratio': [0.01,0.1,0.3,0.5,0.8] #L1 regularization parameter space\n}\n\n# Instantiate grid search using GradientBoostingRegressor\ngrid_elastic = GridSearchCV(estimator = elastic, # regressor we want to use\n                       param_grid = params_elastic, # Hyperparameter space\n                       scoring ='neg_mean_squared_error',\n                       cv = 3, # No of folds\n                       verbose = 1,\n                       n_jobs = -1) # use all cores\n\n# fit the model\ngrid_elastic.fit(housing_prepared,housing_labels)")


# In[ ]:


# Lets look at the Cross Validation score for ElasticNet
print('CV Score for best ElasticNet model: {:.2f}'.format(cv_score(grid_elastic.best_score_)))


# In[ ]:


# Store the best model 
best_model_elastic = grid_elastic.best_estimator_


# ### 3.5 SVM

# In[ ]:


get_ipython().run_cell_magic('time', '', "from sklearn.svm import SVR\n\n# Instantiate the SVM Regressor\nsvr = SVR()\n\n# Creating Parameter grid for GridSearch CV\nparams_svr = {\n    'kernel': ['poly'], # we want a polynomial kernel\n    'degree': [5,8], # degrees to test\n    'gamma':[0.01,0.05], \n    'epsilon': [1.5,3], \n    'coef0':[3,5], # since we are selecting a polynomial kernel\n    'C': [10,30], # Penalty parameter\n    'tol':[1e-7,1e-5] # Tolerance for stopping\n}\n\n# Instantiate grid search using GradientBoostingRegressor\ngrid_svr = GridSearchCV(estimator = svr, # regressor we want to use\n                       param_grid = params_svr, # Hyperparameter space\n                       scoring ='neg_mean_squared_error',\n                       cv = 3, # No. of folds\n                       verbose = 1,\n                       n_jobs = -1) # use all cores\n\n# fit the model\ngrid_svr.fit(housing_prepared,housing_labels)")


# In[ ]:


# Lets look at the Cross Validation score for SVM
print('CV Score for best SVM model: {:.2f}'.format(cv_score(grid_svr.best_score_)))


# In[ ]:


# Store the best model 
best_model_svr = grid_svr.best_estimator_


# ### 3.5 XGBoost

# In[ ]:


get_ipython().run_cell_magic('time', '', "import xgboost as xgb\n\n# Instantiate the SVM Regressor\nxgbr = xgb.XGBRegressor(learning_rate=0.01,objective='reg:linear',booster='gbtree')\n\n# Creating Parameter grid for GridSearch CV\nparams_xgb = {\n    'n_estimators': [8000,10000], #4000,12000\n    'max_depth': [2],\n    'gamma':[0.1,0.2], # Minimum loss reduction to create new tree split ,0.5,0.9\n    'subsample':[0.7], \n    'reg_lambda':[0.1], \n    'reg_alpha':[0.1,0.8] \n}\n\n# Instantiate grid search using GradientBoostingRegressor\ngrid_xgb = GridSearchCV(estimator = xgbr, # regressor we want to use\n                       param_grid = params_xgb, # Hyperparameter space\n                       scoring ='neg_mean_squared_error',\n                       cv = 3, # No. of folds\n                       verbose = 1,\n                       n_jobs = -1) # use all cores\n\n# fit the model\ngrid_xgb.fit(housing_prepared,housing_labels)")


# In[ ]:


# Lets look at the Cross Validation score for XGBRegressor
print('CV Score for best XGBRegressor model: {:.2f}'.format(cv_score(grid_xgb.best_score_)))


# In[ ]:


# Store the best model 
best_model_xgb = grid_xgb.best_estimator_


# ### 3.6 Ensemble

# In[ ]:


# importing Voting Regressor
from sklearn.ensemble import VotingRegressor

# Instantiate the Regressor
voting_reg = VotingRegressor(
    estimators=[('rf', best_model_RF), ('gbr', best_model_GBR), ('elastic', best_model_elastic),
               ('ridge',best_model_ridge),('svr',best_model_svr),('xgb',best_model_xgb)])


# In[ ]:


# Importing cross validation module
from sklearn.model_selection import cross_val_score

# Calculate cross validation score for the Voting regresssor
scores = cross_val_score(voting_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)


# In[ ]:


# Given that we have negative MSE score, lets first get the root squares to get RMSE's and then calculate the mean
voting_reg_score = np.sqrt(-scores)

# Calc mean for RMSE
print(voting_reg_score.mean())


# In[ ]:


# Fitting the voting regressor on the entire training dataset
voting_reg.fit(housing_prepared, housing_labels)

# Predict on test set
pred = voting_reg.predict(test_prepared)


# In[ ]:


# converting to dataframe
preds_df = pd.DataFrame({'Id':test_df_IDs,'SalePrice':pred},index=None)


# In[ ]:


# looking at the first 5 rows
preds_df.head()


# In[ ]:


# Submitting the predictions
preds_df.to_csv('submissions.csv',index=False)


# # Resources:
# 
# Kernels: The following Kernels have been immensely helpful and shaped my approach towards the problem! 
# 
# 1. [Lavanya Shukla's Kernel](https://www.kaggle.com/lavanyashukla01/how-i-made-top-0-3-on-a-kaggle-competition)
# 
# 2. [Serigne's Stacked Regressions Kernel](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard)
# 
# 3. [Pedro Marcelino's EDA Kernel](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python)
# 
# 
# Other online resoruces:
#     
# 1. [Rotating labels:](https://www.drawingfromdata.com/how-to-rotate-axis-labels-in-seaborn-and-matplotlib)
# 
# 2. [Getting feature names after one-hot encoder:](https://stackoverflow.com/questions/54646709/sklearn-pipeline-get-feature-name-after-onehotencode-in-columntransformer/54648023)
# 
# 3. [Random Forest Vs Gradient boosting:](https://stats.stackexchange.com/questions/173390/gradient-boosting-tree-vs-random-forest)
