#!/usr/bin/env python
# coding: utf-8

# ![](http://realsta.in/wp-content/uploads/2016/01/best-home-house-builder-civil-contractor-delhi-gurgaon-india-maxwell-builder.jpg)

#    **A Random Forest Algorithm & Machine Learning Workflow of House Price**

#       
# Being a part of Kaggle gives me unlimited access to learn, share and grow as a Data Scientist. In this kernel, I want to solve Home Price Prediction competition, a popular machine learning dataset for beginners. I am going to share how I work with a dataset step by step from data preparation, data analysis and implementing machine learning models. I will also describe the model results along with many other tips. So let's get started.
# 
# If there are any recommendations/changes you would like to see in this notebook, please leave a comment at the end of this kernel. 
# If you like this notebook or find this notebook helpful, Please feel free to **UPVOTE** and/or leave a comment.
# 
# You can also Fork and Run this kernel on **Github**
# Stay Tuned for More to Come!!

# # Table of Contents

#  - [Introduction](./House%20price%20kaggle%20final.ipynb#Introduction)
# 
#      - [Importing liabraries](#Import-liabraries)
#      
#      - [About Dataset](#About-dataset)
#  
#  - [Overview and Cleaning Data](#Overview-and-cleaning-data)
#  
#      - [Find outliers](#Find-outliers)
#      
#      - [Delating outliers](#Deleting-Outliers)
#      
#  - [Target Variable](#Target-Variable)
#  
#      - [Normal distribution on target variable](#Normal-distribution-on-target-variable)
#      
#      - [Log transformation](#Log-transformation)
#      
#  - [Feature Engineering](#Feature-Engineering)
#  
#      - [Combine train and test data](#Combine-train-&-test-data)
#      
#      - [Use fast.ai liabrary function](#Use-fast.ai-liabrary-function)
#      
#      - [Drop unnecessary features](#Drop-unnecessary-features)
#      
#      - [split train  and test data](#Split-the-train-and-test-data)
#  
#  - [Build the model](#Built-the-model)
#  
#      - [Writting necessary functions for shortcut](#Writting-necessary-functions-for-shortcut)
#      
#      - [Tuning hyperparameter](#Tuning-hyperparameters)
#      
#      - [Final model](#Final-model)
#      
#  - [Model tested on test data](#Model-tested-on-test-data)
#  
#      - [Tested on test data](#Tested-on-test-data)
#      
#      - [Build submission file](#Build-submission-file)
# 
#  - [Credits](#Credits)

# # Introduction
# 

# I learned a lot of things from my first kernel and I treid not to repeat the mistakes in this kernel. I am excited to share my second kernel with the Kaggle community, and I think my journey of data science can leap from this community. please leave a comment if you have any suggestions to make it better!! 
# 
# Going back to the topic of this kernel, I am using visualizations to explain the data, and the  random forest machine learning algorithm used to build model and predict the home prices.

# ## Import liabraries

# Python is an amazing language with many libraries. I am going to import the nessary library as we go on.
# 
# 

# In[ ]:


get_ipython().system('pip install git+https://github.com/fastai/fastai@2e1ccb58121dc648751e2109fc0fbf6925aa8887')
get_ipython().system('apt update && apt install -y libsm6 libxext6')


# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

get_ipython().run_line_magic('matplotlib', 'inline')


# I am using deep learning liabrary named as fast.ai for this kernel. Fast.ai provide special functions like train_cats, proc_df, etc. The train_cats function change any columns of strings in a panda's dataframe to a column of categorical value and proc_df function converts null values with median as well as separate target variabe from training data.

# In[ ]:


from fastai.imports import *
from fastai.structured import train_cats, proc_df, rf_feat_importance
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor

from scipy import stats
from scipy.stats import norm, skew #for some statistics
from sklearn import metrics


# In[ ]:


import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')


# ## About dataset

# In given data there are total 81 features of home and using these features you are going to predict the home prices.
# 
# [Here](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) you can find description of all the features. I have decided to delete Id feature from the data because it is not important for prediction.

# In[ ]:


# !ls ../input


# In[ ]:


df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


df_train.head()


# In[ ]:


df_train.drop('Id', axis=1, inplace=True)
df_test.drop('Id', axis=1, inplace=True)


# In[ ]:


df_train.head()


# # Overview and cleaning data

# ## Find outliers

# **In statistics, an outlier is an observation point that is distant from other observations.**
# 
# The above definition suggests that outlier is something which is separate/different from the crowd
# 

# In[ ]:


fig ,ax = plt.subplots()
ax.scatter(x=df_train['GrLivArea'],y = df_train['SalePrice'])
plt.title('Showing Outliers')
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.show()


# We can see at the bottom right two points with extremely large GrLivArea that are of a low price. These values are huge oultliers.
# 
# That's why I decided the delete them from data.

# ## Deleting Outliers

#  After deleting outliers from the data scatter plot showing the data looks like.

# In[ ]:


#Deleting outliers
df_train.drop(df_train[(df_train['GrLivArea'] > 4000) & (df_train['SalePrice'] < 200000)].index, inplace=True)


# In[ ]:


# check the graph again
fig ,ax = plt.subplots()
ax.scatter(x = df_train['GrLivArea'], y = df_train['SalePrice'])
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.show()


# # Target Variable

# ## Normal distribution on target variable

# We can see the normal distribution on target variable, the graph shows the distribution is right skewed and QQ-plot shows the how hyperline fits on given data.

# In[ ]:


sns.distplot(df_train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(df_train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

# #Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
plt.show()

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
plt.show()


# ## Log transformation

# The hyperline is not fitted on the given data because values of target variable is too high, overcome this problem by using transformation.That's why I'm applying log transformation.
# 
# After transformation, we can see how hyperline fits on target variable.

# In[ ]:


df_train['SalePrice'].head()


# In[ ]:


df_train["SalePrice"] = np.log(df_train["SalePrice"])

sns.distplot(df_train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(df_train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

# #Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
plt.show()

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
plt.show()


# ### Box-Cox Transformation
# 

# This transformation is same like log transformation.

# In[ ]:


# from scipy.special import boxcox1p


# In[ ]:


# df_train['SalePrice'] = boxcox1p(df_train['SalePrice'], 0.15)


# In[ ]:


# #df_train["SalePrice"] = np.log(df_train["SalePrice"])

# sns.distplot(df_train['SalePrice'] , fit=norm);

# # Get the fitted parameters used by the function
# (mu, sigma) = norm.fit(df_train['SalePrice'])
# print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

# # #Now plot the distribution
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
# plt.ylabel('Frequency')
# plt.title('SalePrice distribution')
# plt.show()

# #Get also the QQ-plot
# fig = plt.figure()
# res = stats.probplot(df_train['SalePrice'], plot=plt)
# plt.show()


# # Feature Engineering

# ## Combine train & test data

# We need to anaylse both training and testing data. There is no need of separate analysis on training as well as testing data. We can combine both train and test data and perform analysis on it.
# 
# We are able to split train and test data in given format after complete analysis on combined data. 
# 
# 

# In[ ]:


n_train = df_train.shape[0]
n_test = df_test.shape[0]
y = df_train.SalePrice.values
all_data = pd.concat((df_train, df_test)).reset_index(drop=True)
all_data.drop('SalePrice', axis=1, inplace=True)
print("Size of all data : {}".format(all_data.shape))


# In[ ]:


all_data.head()


# ## Use fast.ai liabrary function

# We can see the how fast.ai liabrary function work on data. 
# 
# The train_cats function change any columns of strings in a panda's dataframe to a column of categorical values. 
# 
# proc_df takes a data frame and splits off the response variable, and changes the dataframe into an entirely numeric dataframe or they can do more than that you can see the fast.ai code on [github](https://github.com/fastai/fastai).

# In[ ]:


train_cats(all_data)


# The proc_df function separate the responce variable but I'm passing other unnecessary feature(Alley) beacause I'm already separate response variable.
# 
# One more reason for not passing response variable is I have used combined data (train & test data for analysis) and in test data response variable is not present.
# 
# If you perform analysis on only training data then you are able to pass the response variable to the proc_df function.

# In[ ]:


all_data,Alley,nas=proc_df(all_data,'Alley')


# In[ ]:


all_data.head()


# ## Drop unnecessary features

# In[ ]:


all_data.columns


# After applying proc_df many na columns are created and this are not useful when training model that's why deleting all na columns from dataframe 

# In[ ]:


dd=['BsmtFinSF1_na', 'BsmtFinSF2_na', 'BsmtFullBath_na', 'BsmtHalfBath_na', 'BsmtUnfSF_na', 'GarageArea_na',
       'GarageCars_na', 'GarageYrBlt_na', 'LotFrontage_na', 'MasVnrArea_na', 'TotalBsmtSF_na']


# In[ ]:


all_data.drop(dd, axis=1, inplace=True)


# In[ ]:


all_data.shape


# In[ ]:


all_data.head()


# After observing all features of dataframe decided to create one new feature named as **TotalSF**. The **TotalSF** is created by adding **1stFlrSF**, **2ndFlrSF** and **TotalBsmtSF** and added this feature into original dataframe.

# In[ ]:


all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']


# ## Split the train and test data

# Separate the training and testing data and build a model on training data.

# In[ ]:


train = all_data[:n_train]
test = all_data[n_train:]


# In[ ]:


train.head()


# In[ ]:


test.head()


# # Built the model

# ## Writting necessary functions for shortcut

# Write some function which are helpful in calculations, **rmse, print_score, split_vals** function are created. 
# 
# The rmse calculate **Root Mean Square Error** for the given data, print_score calculate the rmse of training and testing data and accuracy of training and testing data.
# 
# The data is already randomised hence we don't need to use random split.

# In[ ]:


def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train),y_train), rmse(m.predict(X_valid),y_valid),
           m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# In[ ]:


def split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_valid = 88
n_trn = len(train) - n_valid
X_train, X_valid = split_vals(train, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape


# ## Tuning hyperparameters

# After applying feature enginering we are able to build model on data. 
# 
# Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random decision forests correct for decision trees' habit of overfitting to their training set.
# 
# Random Forest Regressor.
# 
# A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
# 
# According to Random Forest
# definition it is fit for building model on given data. 
# 
# So, Let start to build random forest model!

# Using all features build random forest model with two parameters and print accuracy of training and testing data by calling print_score function.
# 

# In[ ]:


m = RandomForestRegressor(n_jobs=-1, random_state=1)
m.fit(X_train, y_train)
print_score(m)


# Build the following model using 6 parameters and see result in the output. We can see in this model I'm using maximum 50% features, build 46 decision trees, keep the maximum depth of each decision tree is 10. 
# 
# Out-of-bag (OOB) error, also called out-of-bag estimate, is a method of measuring the prediction error of random forests
# 
# OOB_score defines score of the training dataset obtained using an out-of-bag estimate.
# 
# There is small change in this model by comparing with previous model.

# In[ ]:


m = RandomForestRegressor(n_jobs=-1, random_state=2, n_estimators=46, oob_score=True, max_features=0.5, 
                          max_depth=10)
m.fit(X_train,y_train)
print_score(m)


# We can see the changes values of n_estimators, max_features and decreases OOB_score of the model. 

# In[ ]:


m = RandomForestRegressor(n_jobs=-1, random_state=1, n_estimators=25, oob_score=True, max_features=0.6)
m.fit(X_train,y_train)
print_score(m)


# If increasing trees in the forest as well as increases max_features, there is small change in OOB_score but first model is better than this model.

# In[ ]:


m = RandomForestRegressor(n_jobs=-1, random_state=1, n_estimators=40, oob_score=True, max_features=0.6, 
                          )
m.fit(X_train,y_train)
print_score(m)


# In this model added one new parameter named as min_samples_leaf. It means at the end leaf of tree contains minimum 3 data points.
# 
# Check small change in OOB_score of this model compare with previous model. 

# In[ ]:


m = RandomForestRegressor(n_jobs=-1, random_state=6, n_estimators=60, oob_score=True, max_features=0.5, 
                          max_depth=16, min_samples_leaf=3)
m.fit(X_train,y_train)
print_score(m)


# This model is not good while comparing with old model because in this model reduces OOB_score while tuning hyperparameter.
# 
# Added max_leaf_nodes parameter in this model this represents maximum 450 leaf nodes are available in the tree.

# In[ ]:


m = RandomForestRegressor(n_jobs=-1, random_state=6, n_estimators=60, oob_score=True, max_features=0.4, 
                          max_depth=16, min_samples_leaf=3, max_leaf_nodes=450)
m.fit(X_train,y_train)
print_score(m)


# This model improve the OOB_score as compared with previous model. The **OOB_score** of current model is **0.8908** and previous model OOB_score is **0.8862**.
# 
# In this model added one new parameter named as min_impurity_decrease. This parameter a node will be split if this split induces a decrease of the impurity greater than or equal to this value

# In[ ]:


m = RandomForestRegressor(n_jobs=-1, random_state=10, n_estimators=100, oob_score=True, max_features=0.5, 
                          max_depth=14, min_samples_leaf=3, max_leaf_nodes=400, min_impurity_decrease=0.00001)
m.fit(X_train,y_train)
print_score(m)


# Again improvement in the next model as compared with previous model. 
# 
# We can see OOB_score is increased while tuning parameters. The **OOB_score** of current model is **0.8928** while **OOB_score** of previous model is **.8908**

# In[ ]:


m = RandomForestRegressor(n_jobs=-1, random_state=10, n_estimators=160, oob_score=True, max_features=0.5, 
                          max_depth=14, min_samples_leaf=2, max_leaf_nodes=400, min_impurity_decrease=0.00001)
m.fit(X_train,y_train)
print_score(m)


# ## Final model

# **Finally, I give my best effort to build a random forest model** using number of parameters of random forest.
# 
# For this model I have tuned every parameter of the RandomForestRegressor and finally represent model with **OOB_score = 0.89291**. 

# In[ ]:


m = RandomForestRegressor(n_jobs=-1, random_state=10, n_estimators=160, oob_score=True, max_features=0.5, 
                          max_depth=None, min_samples_leaf=2, max_leaf_nodes=250, min_impurity_decrease=0.00001,
                          min_impurity_split=None)

m.fit(X_train,y_train)
print_score(m)


# # Model tested on test data

# ## Tested on test data

# In[ ]:


SalePrice = m.predict(test)


# In[ ]:


df_sample = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


df_sample.head()


# ## Build submission file

# In[ ]:


SalePrice = np.exp(SalePrice)


# In[ ]:


df_sample['SalePrice'] = SalePrice


# In[ ]:


df_sample.head()


# In[ ]:


df_sample.to_csv('Home_price.csv', columns=['Id','SalePrice'], index=False)


# In[ ]:


df_sample.SalePrice.head()


# # Credits
# 

# - To **[fast.ai](https://www.fast.ai/)** where I started my deep learning journey.
# - To **[Palash Karmore](https://www.linkedin.com/in/palash-karmore?originalSubdomain=in)**, for motivating as well as give suggestions on my mistake when I'm creating in my data science   journey.  

# # Resources

# Here are some of the links I found helpful while writing kernel
# 
# - [Stacked Regressions to predict House Prices](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard) from this kernel I have learned Box-Cox transformation as well as how combine train and test data and perform analysis.
