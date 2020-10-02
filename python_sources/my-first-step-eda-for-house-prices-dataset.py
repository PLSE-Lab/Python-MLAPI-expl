#!/usr/bin/env python
# coding: utf-8

# # Overview
# This is my first Kernel commit to kaggle community. The objectives of this Kernel is the following two points;
# - exploratory data analysis (checking distributions and finding useful variables for prediction)
# - applying machine learning algorithms
# 
# ## Task: House Prises Dataset
# The task is to predict the variable named as ``SalePrice``, which indicates the saled price of estates, by the following independent variables. It should be noted that measurement scales of these variables are mixed; numerical and categorical.
# - categorical variables
#     -  ``MSZoning``, ``Street``,  ``Alley``, ,  ``LotShape``,   ``LandContour``,  ``Utilities``,  ``LotConfig``,  ``LandSlope``,  ``Neighbourhood``,  ``Condition1``,   ``Condition2``,  ``BldgType``,  ``HouseStyle``,  ``RoofStyle``,  ``RoofMatl``,  ``Exterior1st``,  ``Exterior2nd``,   ``MasVnrType``,  ``MasVnrArea``,  ``ExterQual``,  ``ExterCond``,  ``Foundation``,  ``BsmtQual``,  ``BsmtCond``,  ``BsmtExposure``,  ``BsmtFinType1``,  ``BsmtFinType2``,  ``Heating``,  ``HeatingQC``,  ``CentralAir``,  ``Electrical``,  ``KitchenQual``,  ``Functional``,  ``FireplaceQu``,  ``GarageFinish``,  ``GarageQual``,  ``GarageCond``,  ``PavedDrive``,  ``PoolQC``,  ``Fence``,  ``MiscFeature``,  ``SaleCondition``
# - numerical variables
#     - ``MSSubClass``,  ``LotFrontage``,  ``LotArea``,  ``OVerallQual``,  ``OverallCond``,  ``YearBuilt``,  ``YearRemodAdd``,  ``MasVnrArea``,  ``BsmtFinSF1``,  ``BsmtFinSF2``,  ``BsmtUnfSF``,  ``TotalBsmtSF``,  ``1stFlrSF``,  ``2ndFlrSF``,  ``LowQualFinSF``,  ``FrLivArea``,  ``BsmtFullBath``,  ``BsmtHalfBath``,  ``FullBath``,  ``HalfBath``,  ``BedroomAbvGr``,  ``KitchenAbvGr``,  ``TotRmsAbvGrd``,  ``Fireplaces``,  ``GarageYrBlt``,  ``GarageCars``,  ``GarageArea``,  ``WoodDeckSF``,  ``OpenPorchSF``,  ``EnclosedPorc``,  ``3SsnPorch``,  ``ScreenPorch``,  ``PoolArea``,  ``MiscVal``,  ``MSSold``,   ``YrSold``,  ``YrSold``

# # Section1: Exploratory Data Analysis
# ## Importing dataset
# ``train.csv`` is imported via ``pd.read_csv`` after importing some useful modules.

# In[2]:


import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')

dat = pd.read_csv("../input/train.csv")
dat = dat.drop("Id",axis=1) #Id is dropped

print("number of observations:{}".format(dat.shape[0]))
print("number of variables:{}".format(dat.shape[1]))


# Wow, the dataset contains 1460 observations and 80 variables (including 1 dependent variable).

# ## Distribution of dependent variable
# It is important to check empirical distribution of dependent variable, before applying models.
# It seems ``SalePrice`` variable is distributed with skewed normal distribution, and therefore log-transformed one can be considered to be normally distributed.

# In[3]:


fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,7.5))
ax1=sns.distplot(dat.SalePrice,ax=ax1)
ax1.set_title("raw data")
ax2=sns.distplot(np.log(dat.SalePrice),ax=ax2)
ax2.set_title("log-transformed")


# ## Visualizing correlation structure (only numerical variables)
# Correlation matrix between dependent/independent variables is visualized as a heatmap.

# In[4]:


corr_mat = dat.corr()
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr_mat, vmax=.8, square=True)


# There are some variables which are highly correlated with ``SalePrice`` (bottom of the heatmap), which are considered to be inportant for prediction.
# Also, there are some pairs of variables which are highly correlated, and we have to care such variable when using simple regression model (multi-colinearity).
# Scatter plots of ``SalePrice`` and highly correlated variables are shown below.

# In[5]:


fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2,ncols=3,figsize=(12,8))
#fig, (ax1, ax2, ax3) = plt.subplots(ncols=3,figsize=(12,8))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
ax1.scatter(dat.OverallQual,dat.SalePrice)
ax1.set_title("OverallQual")
ax2.scatter(dat.YearBuilt,dat.SalePrice)
ax2.set_title("YearBuilt")
ax3.scatter(dat.FullBath,dat.SalePrice)
ax3.set_title("FullBath")
ax4.scatter(dat.TotRmsAbvGrd,dat.SalePrice)
ax4.set_title("TotRmsAbvGrd")
ax5.scatter(dat.GarageCars,dat.SalePrice)
ax5.set_title("GarageCars")
ax6.scatter(dat.GarageArea,dat.SalePrice)
ax6.set_title("GarageArea")


# It seems natural that these variables are positively correlated with ``SalePrice``, since they represent quality and area of estates. 
# Also, it is interesting to see that ``YearBuild`` and ``SalePrice`` are positively correlated, but there exists some outliers for estates built before 1900. They must be historic architectures such as castle, etc.

# ## Checking categorical variables
# Next, I've checked relationships between ``SalePrice`` and some categorical variables.
# As well as area of estates, environment around estates is important for price.

# In[6]:


dat_sub = pd.concat([dat['SalePrice'], dat["Neighborhood"]], axis=1)
f, ax = plt.subplots(figsize=(16, 5))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
sns.boxplot(x="Neighborhood", y="SalePrice", data=dat_sub)
plt.xticks(rotation=90)

dat_sub = pd.concat([dat['SalePrice'], dat["Street"]], axis=1)
f, ax = plt.subplots(figsize=(16, 5))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
sns.boxplot(x="Street", y="SalePrice", data=dat_sub)
plt.xticks(rotation=90)

dat_sub = pd.concat([dat['SalePrice'], dat["HouseStyle"]], axis=1)
f, ax = plt.subplots(figsize=(16, 5))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
sns.boxplot(x="HouseStyle", y="SalePrice", data=dat_sub)
plt.xticks(rotation=90)



# ## Treating Missing Values
# There are some variables having missing values. These variables are easily found by the following code. For example, over 99% of ``PoolQC`` value contains missing values (probably because most of the estates don't have pool). In the following analysis, I've omitted these variables with nulls.

# In[7]:


total_nulls = dat.isnull().sum().sort_values(ascending=False)
percent_nulls = (dat.isnull().sum()/dat.isnull().count()).sort_values(ascending=False)
dat_missing = pd.concat([total_nulls, percent_nulls], axis=1, keys=['Total', 'Percent'])
dat_missing[dat_missing["Percent"]>0]


# In[8]:


dat_comp = dat.drop((dat_missing[dat_missing["Percent"] > 0]).index,1)


# ## Dummy coding for categorical variables
# Categorical variables were coded into dummy variable, which was simply accomlished as follows;

# In[9]:


dat_comp = pd.get_dummies(dat_comp)
dat_comp.head()


# # Section2: Applying Machine Learning Models
# After some visualization of the training dataset and preprocessing, I've proceed to applying some ML models to the dataset.
# Here, the following regression models were considered, which are available in scikit-learn;
# - LASSO regression
# - Ridge regression
# - Random Forest regression
# - Adaboost regression
# 
# Some tuning parameters for the models were tuned by Grid Search function. It should be noted that we must not use simple linear regression, because of multi-colinearity found in the previous section.
# 
# The procedure for applying ML model is as follows;
# - Split ``dat_comp`` into training (90%) and test (10%) set.
# - Apply grid search for tuning some parameters.
# - Fit the model with tuned parameters to the training set.
# - Evaluate accuracy for test set.

# ## Splitting dataset into training/test set

# In[10]:


from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(dat_comp.drop("SalePrice", axis=1), dat_comp.SalePrice, random_state=1234,train_size=0.9)


# ## LASSO regression
# LASSO is a kind of penalized regression model and it shrinks regression coefficients of some variables toword zero, which are consided to be unnecesary for prediction.
# LASSO is also known to produce stable estimate even if multi-colinearity exists.
# 
# The resulting R-squared score was 0.831, which indicates 83% of total variance of dependent variable is explained.

# In[17]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

param = {
    "alpha": np.linspace(0, 1000, 100)
}

model_lasso = Lasso()
tune_lasso = GridSearchCV(model_lasso, param,n_jobs=-1)
tune_lasso.fit(train_X, train_y)


# In[18]:


best_lasso = tune_lasso.best_params_
print(tune_lasso.best_estimator_)
print("Validation accuracy (LASSO): {}".format(tune_lasso.score(test_X,test_y)))


# Scatter plot of predicted/observed ``SalePrice`` values. It seems LASSO performs quite well except for some outliers.

# In[20]:


plt.scatter(tune_lasso.predict(train_X),train_y)
plt.xlabel("predict")
plt.ylabel("observed")
plt.show()


# ## Ridge regression
# Ridge regression is also known as a member of penalty regression family, but Ridge estimate does not contains exact zero elements while LASSO estimates contains some elements equaling to zero.

# In[29]:


from sklearn.linear_model import Ridge
param = {
    "alpha": np.linspace(0, 1000, 100)
}

model_ridge = Ridge()
tune_ridge = GridSearchCV(model_ridge, param,n_jobs=-1)
tune_ridge.fit(train_X, train_y)


# In[31]:


best_ridge = tune_ridge.best_params_
print("Validation accuracy (LASSO): {}".format(tune_ridge.score(test_X,test_y)))


# Scatter plot of predicted/observed ``SalePrice`` values for Ridge regression.

# In[32]:


plt.scatter(tune_ridge.predict(train_X),train_y)
plt.xlabel("predict")
plt.ylabel("observed")
plt.show()


# ## Random Forest regression
# Random Forest regression is an ensemble model, in which some weak regressor called trees are combined into a strong regressor.

# In[102]:


from sklearn.ensemble import RandomForestRegressor

param = {
    "max_features": ["auto", "sqrt", "log2"],
    "max_depth": [10,20,30],
    "random_state": [i for i in range(20)]
}

model_rf = RandomForestRegressor()
tune_rf = GridSearchCV(model_rf, param,n_jobs=-1)
tune_rf.fit(train_X, train_y)


# In[105]:


best_rf = tune_rf.best_params_
print(tune_rf.best_estimator_)
print("Validation accuracy (RandomForest): {}".format(tune_rf.score(test_X,test_y)))


# Scatter plot of predicted/observed SalePrice values for Random Forest regression.

# In[106]:


plt.scatter(tune_rf.predict(train_X),train_y)
plt.xlabel("predict")
plt.ylabel("observed")
plt.show()


# ## Adaboost regression
# Adaboost is a kind of boosting technique. Sorry I still don't know the detail about the model.

# In[107]:


from sklearn.ensemble import AdaBoostRegressor

param = {
    "loss": ["linear", "square", "exponential"],
    "learning_rate": np.linspace(0.1, 1, 10),
    "n_estimators": [10,20,30]
}

model_ab = AdaBoostRegressor()
tune_ab = GridSearchCV(model_ab, param,n_jobs=-1)
tune_ab.fit(train_X, train_y)


# In[108]:


best_ab = tune_ab.best_params_
print(tune_ab.best_estimator_)
print("Validation accuracy (Adaboost): {}".format(tune_ab.score(test_X,test_y)))


# Scatter plot of predicted/observed SalePrice values for Adaboost regression.

# In[109]:


plt.scatter(tune_ab.predict(train_X),train_y)
plt.xlabel("predict")
plt.ylabel("observed")
plt.show()


# ## Variable importance in RandomForest regression
# The best model, RandomForest regression, was specified above the trials. For the model variable importance which indicates important variables for prediction is available. 
# The top 10 important variables overall implies 
# - If the quality of the estate is good its price is also high
#     - can be seen at ``OverallQual`` and ``YearBuilt``
# - The larger the house price is also high
#     - can be seen at ``GarageCars`` and ``1stFlrSF`` etc.
#     
# For me, it makes sense because we prefer high quality and large house, not poor quality and old estate.

# In[110]:


feat_imp = tune_rf.best_estimator_.feature_importances_
feat_imp = pd.DataFrame({
                    "variable":train_X.columns,
                    "importance":feat_imp
})

feat_imp = feat_imp.sort_values("importance",ascending=False)
feat_imp.head(10)


# # Submission and FutureWork
# The trained RandomForest regressor model was applied to the test dataset. 
# My score was 0.16787. Although it is a poor result, but I believe it is a good first step for me.
# 
# For the future work, I'm going to consider the following points;
# - extending grid search area for tuning parameters of ML models
# - feature engineering
# - log transformation of dependent variable
# 

# In[111]:


dat_test = pd.read_csv("../input/test.csv")
Ids = dat_test["Id"]
dat_test = dat_test.drop("Id",axis=1) #Id is dropped

#preprocessing for test dataset
dat_test_comp = dat_test.drop((dat_missing[dat_missing["Percent"] > 0]).index,1)
dat_test_comp = pd.get_dummies(dat_test_comp)
dat_test_comp.fillna(method='ffill',inplace=True)

#add missing categories as dummy variables for test set
notintest_vals = set(dat_comp)-set(dat_test_comp)
for val in notintest_vals: 
    if val != "SalePrice":
        dat_test_comp[val] = np.zeros(dat_test_comp.shape[0])
        
#apply RF regression
pred_test = tune_rf.predict(dat_test_comp)
pred_table = pd.DataFrame({
    "Id": Ids,
    "SalePrice": pred_test
})
pred_table.to_csv("predict.csv",index=False)


# ### Acknowledgement
# For the kernel, I've used and modied some codes of [Comprehensive data exploration with Python](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python/code) by Pedro Marcelino.

# In[ ]:




