#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import some necessary librairies

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


from scipy import stats
from scipy.stats import norm, skew #for some statistics

plt.style.use('ggplot')
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8")) #check the files available in the directory


# In[ ]:


#Now let's import and put the train and test datasets in  pandas dataframe

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


##display the first five rows of the train dataset.
train.head(5)


# In[ ]:


##display the first five rows of the test dataset.
test.head(5)


# In[ ]:


#check the numbers of samples and features
print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))

#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

#check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test.shape))


# #Data Processing

# Let's explore these outliers
# 

# We can see at the bottom right two with extremely large GrLivArea that are of a low price. These values are huge oultliers.
# Therefore, we can safely delete them.

# In[ ]:


areas = ['GrLivArea', 'GarageArea', 'TotalBsmtSF']

for columns in areas:
  plt.figure()
  sns.lmplot(y = 'SalePrice', x = columns, data = train);


# In[ ]:


#Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

#Check the graphic again
sns.lmplot(x = 'GrLivArea', y = 'SalePrice',data = train)
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# #### We can see there are few outliers and we will deal with them later.
# 
# #### Let us see how the SalePrice is related categorical variables like Overall quality and OverallCond of the plot
# 
# ###Note : 
#  Outliers removal is note always safe.  We decided to delete these two as they are very huge and  really  bad ( extremely large areas for very low  prices). 
# 
# There are probably others outliers in the training data.   However, removing all them  may affect badly our models if ever there were also  outliers  in the test data. That's why , instead of removing them all, we will just manage to make some of our  models robust on them. You can refer to  the modelling part of this notebook for that. 

# In[ ]:


plt.figure(figsize=(10,6))

sns.boxplot(x = 'OverallQual', y= 'SalePrice',data = train)
plt.show()


# In[ ]:


plt.figure(figsize=(10,6))
sns.boxplot(x = 'OverallCond', y = 'SalePrice', data = train)
plt.show();


# #### We can see that with better house conditions the prices increase. 
# 
# #### Let us see how the house prices have changed over the years

# In[ ]:


plt.figure(figsize=(21,12))

ax = sns.boxplot(x = 'YearBuilt', y = 'SalePrice', data = train)
# ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, ha = 'right', fondsize = 12)
plt.xticks(rotation =90)
plt.tight_layout()
plt.show()


# #### We can see the house prices have increased over the years and though a boxplot we can have an understanding of their distributions.
# 
# ### Let us find the correlation between the variables

# In[ ]:


corr = train.corr()

plt.figure(figsize=(10,10))
k = 10 #number of variables for heatmap
cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True,yticklabels=cols.values, xticklabels=cols.values)


# ##Target Variable
# 
# **SalePrice** is the variable we need to predict. So let's do some analysis on this variable first.

# In[ ]:


sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()


# The target variable is right skewed.  As (linear) models love normally distributed data , we need to transform this variable and make it more normally distributed.

#  **Log-transformation of the target variable**

# In[ ]:


#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train["SalePrice"] = np.log1p(train["SalePrice"])

#Check the new distribution 
sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()


# 

# 1. The skew seems now corrected and the data appears more normally distributed. 
# 
# ## Features engineering
# 
# let's first  concatenate the train and test data in the same dataframe

# In[ ]:


ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))


# ## Let's deal with missing data

# In[ ]:


NA_values = all_data.isna().sum().sort_values(ascending=False)[:34]

NA = pd.concat([NA_values,(NA_values/len(all_data)*100)],axis=1)

NA


# In[ ]:


plt.figure(figsize=(12,10))
ax = sns.barplot(y = NA.iloc[:,0], x = NA.index)
plt.xticks(rotation = 90)
plt.show()


# ###Imputing missing values 
# 
# We impute them  by proceeding sequentially  through features with missing values 

# In[ ]:


#Similarly if there is no basement then the following values would be 0 
NA_zero = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
           'MasVnrArea']

#Similarly NA indicates no feature here and we hence can fill it with 'None'
Na_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu','GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 
           'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']

for col in NA_zero:
    all_data[col] = all_data[col].fillna(0)

#Similarly None for categorical attributes
for col in Na_none:
    all_data[col] = all_data[col].fillna('None')

    
#Since area of a street is connected to the house property we can fill in missing values by the median LotFrontage of the neighborhood.
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
    
#Here 'RL' is the most common value and we can fill this in using mode
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

# For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA so it is safe to drop it
all_data = all_data.drop(['Utilities'], axis=1)

# Data description says NA means typical
all_data["Functional"] = all_data["Functional"].fillna("Typ")

#It has one NA value. and is mostly 'SBrkr', we can use mode imputation for the missing value
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

#Just like electrical it has one missing value
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

#Same as above
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

#Na most likely means No building class
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")


# #### Let us see if there are any null values remaining

# In[ ]:


all_data.isnull().sum().sort_values(ascending = False).head()


# It remains no missing value.
# 
# ###More features engeneering
# 
# **Transforming some numerical variables that are really categorical**
# 

# In[ ]:


#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)


# **Label Encoding some categorical variables that may contain information in their ordering set** 

# In[ ]:


from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))


# **Adding one more important feature**
# 
# Since area related features are very important to determine house prices, we add one more feature which is the total area of basement, first and second floor areas of each house

# In[ ]:


# Adding total sqfootage feature 
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']


# **Skewed features**

# In[ ]:


numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)


# **Box Cox Transformation of (highly) skewed features**

# We use the scipy  function boxcox1p which computes the Box-Cox transformation of **\\(1 + x\\)**. 
# 
# Note that setting \\( \lambda = 0 \\) is equivalent to log1p used above for the target variable.  
# 
# See [this page][1] for more details on Box Cox Transformation as well as [the scipy function's page][2]
# [1]: http://onlinestatbook.com/2/transformations/box-cox.html
# [2]: https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.special.boxcox1p.html

# In[ ]:


skewness = skewness[abs(skewness.Skew) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)
    
#all_data[skewed_features] = np.log1p(all_data[skewed_features])


# **Getting dummy categorical features**

# In[ ]:



all_data = pd.get_dummies(all_data)
print(all_data.shape)


# Getting the new train and test sets. 

# In[ ]:


train = all_data[:ntrain]
test = all_data[ntrain:]


# #Modelling
# 
# **Import librairies**

# In[ ]:


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


# **Define a cross validation strategy**
# 
# Cross-validation is primarily used in applied machine learning to estimate the skill of a machine learning model on unseen data.<br> That is, to use a limited sample in order to estimate how the model is expected to perform in general when used to make predictions on data not used during the training of the model.

# In[ ]:


#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# ##Base models

# -  **LASSO  Regression**  : 
# 
# Lasso uses L1 regularization technique <br>
# It is generally used when we have more number of features, because it automatically does feature selection.
# 
# This model may be very sensitive to outliers. So we need to made it more robust on them. For that we use the sklearn's  **Robustscaler()**  method on pipeline 

# In[ ]:


lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))


# - **Elastic Net Regression** :
# 
# Elastic net is basically a combination of both L1 and L2 regularization. So if you know elastic net, you can implement both Ridge and Lasso by tuning the parameters. So it uses both L1 and L2 penality term <br>
# <br>
# We have a bunch of correlated independent variables in a dataset, then elastic net will simply form a group consisting of these correlated variables. Now if any one of the variable of this group is a strong predictor (meaning having a strong relationship with dependent variable), then we will include the entire group in the model building, because omitting other variables (like what we did in lasso) might result in losing some information in terms of interpretation ability, leading to a poor model performance.

# In[ ]:


ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))


# - **Kernel Ridge Regression** :
# 
# Kernel ridge regression is essentially the same as usual ridge regression, but uses the kernel trick to go non-linear.

# In[ ]:


KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)


# - **Gradient Boosting Regression** :
# 
# Boosting is an ensemble technique where new models are added to correct the errors made by existing models. Models are added sequentially until no further improvements can be made.
# <br>
# XGBoost is an implementation of gradient boosted decision trees designed for speed and performance.
# 
# 
# 
# With **huber**  loss that makes it robust to outliers
#     

# In[ ]:


GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)


# - **XGBoost** :

# In[ ]:


model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)


# - **LightGBM** :
# 
# Light GBM is a fast, distributed, high-performance gradient boosting framework based on decision tree algorithm, used for ranking, classification and many other machine learning tasks.

# In[ ]:


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


# ###Base models scores
# 
# Let's see how these base models perform on the data by evaluating the  cross-validation rmsle error

# In[ ]:


score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:



score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))


# ##Stacking  models
# ###Simplest Stacking approach : Averaging base models
# 
# We begin with this simple approach of averaging base models.  We build a new **class**  to extend scikit-learn with our model and also to laverage encapsulation and code reuse ([inheritance][1]) 
# 
# 
#   [1]: https://en.wikipedia.org/wiki/Inheritance_(object-oriented_programming)
#   
#   **Averaged base models class**

# In[ ]:


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   


# **Averaged base models score**
# 
# We just average four models here **ENet, GBoost,  KRR and lasso**.  Of course we could easily add more models in the mix. 

# In[ ]:


averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# Wow ! It seems even the simplest stacking approach really improve the score . This encourages 
# us to go further and explore a less simple stacking approch. 

# ###Less simple Stacking : Adding a Meta-model

# In this approach, we add a meta-model on averaged base models and use the out-of-folds predictions of these base models to train our meta-model. 
# 
# The procedure, for the training part, may be described as follows:
# 
# 
# 1. Split the total training set into two disjoint sets (here **train** and .**holdout** )
# 
# 2. Train several base models on the first part (**train**)
# 
# 3. Test these base models on the second part (**holdout**)
# 
# 4. Use the predictions from 3)  (called  out-of-folds predictions) as the inputs, and the correct responses (target variable) as the outputs  to train a higher level learner called **meta-model**.
# 
# The first three steps are done iteratively . If we take for example a 5-fold stacking , we first split the training data into 5 folds. Then we will do 5 iterations. In each iteration,  we train every base model on 4 folds and predict on the remaining fold (holdout fold). 
# 
# So, we will be sure, after 5 iterations , that the entire data is used to get out-of-folds predictions that we will then use as 
# new feature to train our meta-model in the step 4.
# 
# For the prediction part , We average the predictions of  all base models on the test data  and used them as **meta-features**  on which, the final prediction is done with the meta-model.
# 

# **Stacking averaged Models Class**

# In[ ]:


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)


# **Stacking Averaged models Score**

# To make the two approaches comparable (by using the same number of models) , we just average **Enet KRR and Gboost**, then we add **lasso as meta-model**.

# In[ ]:


stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),
                                                 meta_model = lasso)

score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


# We get again a better score by adding a meta learner

# ## Ensembling StackedRegressor, XGBoost and LightGBM

# We add **XGBoost and LightGBM** to the** StackedRegressor** defined previously. 

# We first define a rmsle evaluation function 

# In[ ]:


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# ###Final Training and Prediction

# **StackedRegressor:**

# In[ ]:


stacked_averaged_models.fit(train.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
print(rmsle(y_train, stacked_train_pred))


# **XGBoost:**

# In[ ]:


model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))


# **LightGBM:**

# In[ ]:


model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print(rmsle(y_train, lgb_train_pred))


# In[ ]:


'''RMSE on the entire Train data when averaging'''

print('RMSLE score on train data:')
print(rmsle(y_train,stacked_train_pred*0.70 +
               xgb_train_pred*0.15 + lgb_train_pred*0.15 ))


# **Ensemble prediction:**

# In[ ]:


ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15


# **Submission**

# In[ ]:


sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = ensemble
sub.to_csv('submission.csv',index=False)


# **If you found this notebook helpful or you just liked it , some upvotes would be very much appreciated**.

# In[ ]:




