#!/usr/bin/env python
# coding: utf-8

# 
# ### Predicting Sale Price of Houses
# 
# The problem at hand aims to predict the final sale price of homes based on different explanatory variables describing aspects of residential homes. Predicting house prices is useful to identify fruitful investments, or to determine whether the price advertised for a house is over or underestimated, before making a buying judgment.
# 
# To download the House Price dataset go this website:
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
# 
# Scroll down to the bottom of the page, and click on the link 'train.csv', and then click the 'download' blue button towards the right of the screen, to download the dataset.
# Save it to a directory of your choice.
# 
# For the Kaggle submission, download also the 'test.csv' file, which is the one we need to score and submit to Kaggle. Rename the file to 'house_price_submission.csv'
# 
# **Note that you need to be logged in to Kaggle in order to download the datasets**.
# 
# If you save it in the same directory from which you are running this notebook and name the file 'houseprice.csv' then you can load it the same way I will load it below.
# 
# ====================================================================================================

# ## House Prices dataset

# In[ ]:


# to handle datasets
import pandas as pd
import numpy as np

# for plotting
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# to divide train and test set
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import MinMaxScaler

# for tree binarisation
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score


# to build the models
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb

# to evaluate the models
from sklearn.metrics import mean_squared_error
from math import sqrt

pd.pandas.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings('ignore')


# ### Load Datasets

# In[ ]:


# load dataset
data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
print(data.shape)
data.head()


# In[ ]:


# Load the dataset for submission (the one on which our model will be evaluated by Kaggle)
# it contains exactly the same variables, but not the target

submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
submission.head()


# The House Price dataset is bigger than the Titanic dataset. It contains more variables for each one of the houses. Thus, manual inspection of each one of them is a bit time demanding. Therefore, here instead of deciding variable by variable what is the best way to proceed, I will try to automate the feature engineering pipeline, making some a priori decisions on when I will apply one technique or the other, and then expanding them to the entire dataset.

# ### Types of variables 
# 
# Let's go ahead and find out what types of variables there are in this dataset

# In[ ]:


# let's inspect the type of variables in pandas
data.dtypes


# There are a mixture of categorical and numerical variables. Numerical are those of type int and float and categorical those of type object.

# In[ ]:


# we also have an Id variable, that we shoulld not use for predictions:

print('Number of House Id labels: ', len(data.Id.unique()))
print('Number of Houses in the Dataset: ', len(data))


# Id is a unique identifier for each of the houses. Thus this is not a variable that we can use.
# 
# #### Find categorical variables

# In[ ]:


# find categorical variables
categorical = [var for var in data.columns if data[var].dtype=='O']
print('There are {} categorical variables'.format(len(categorical)))


# #### Find temporal variables
# 
# There are a few variables in the dataset that are temporal. They indicate the year in which something happened. We shouldn't use these variables straightaway for model building. We should instead transform them to capture some sort of time information. Let's inspect these temporal variables:
# 

# In[ ]:


# make a list of the numerical variables first
numerical = [var for var in data.columns if data[var].dtype!='O']

# list of variables that contain year information
year_vars = [var for var in numerical if 'Yr' in var or 'Year' in var]

year_vars


# In[ ]:


data[year_vars].head()


# We can see that these variables correspond to the years in which the houses were built or remodeled or a garage was built, or the house was indeed sold. It would be better if we captured the time elapsed between the time the house was built and the time the house was sold for example. We are going to do that in the feature engineering section. 
# 
# We have another temporal variable: MoSold, which indicates the month in which the house was sold. Let's inspect if the house price varies with the time of the year in which it is sold:

# In[ ]:


data.groupby('MoSold')['SalePrice'].median().plot()
plt.title('House price variation in the year')
plt.ylabel('mean House price')


# The price seems to vary depending on the time of the year in which the house is sold. This information will be captured when we engineer this variable later on.
# 
# 
# #### Find discrete variables
# 
# To identify discrete variables, I will select from all the numerical ones, those that contain a finite and small number of distinct values. See below.

# In[ ]:


# let's visualise the values of the discrete variables
discrete = []

for var in numerical:
    if len(data[var].unique())<20 and var not in year_vars:
        print(var, ' values: ', data[var].unique())
        discrete.append(var)
print()
print('There are {} discrete variables'.format(len(discrete)))


# #### Continuous variables

# In[ ]:


# find continuous variables
# let's remember to skip the Id variable and the target variable SalePrice, which are both also numerical

numerical = [var for var in numerical if var not in discrete and var not in ['Id', 'SalePrice'] and var not in year_vars]
print('There are {} numerical and continuous variables'.format(len(numerical)))


# Perfect!! Now we have inspected and have a view of the different types of variables that we have in the house price dataset. Let's move on to understand the types of problems that these variables have.
# 
# ### Types of problems within the variables (section 3)
# 
# #### Missing values

# In[ ]:


# let's visualise the percentage of missing values for each variable
for var in data.columns:
    if data[var].isnull().sum()>0:
        print(var, data[var].isnull().mean())


# In[ ]:


# let's now determine how many variables we have with missing information

vars_with_na = [var for var in data.columns if data[var].isnull().sum()>0]
print('Total variables that contain missing information: ', len(vars_with_na))


# There are quite a few variables with missing information. And they differ in the percentage of observations for which information are missing. 
# Let's go ahead and inspect those variables that show missing information for most of their observations.

# In[ ]:


# let's inspect the type of those variables with a lot of missing information
for var in data.columns:
    if data[var].isnull().mean()>0.80:
        print(var, data[var].unique())


# The variables that contain a lot of missing data are categorical variables. We will need to fill those out later in the feature engineering section.
# 
# #### Outliers

# In[ ]:


# let's look at the numerical variables
numerical


# In[ ]:


# let's make boxplots to visualise outliers in the continuous variables 
# and histograms to get an idea of the distribution

for var in numerical:
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    fig = data.boxplot(column=var)
    fig.set_title('')
    fig.set_ylabel(var)
    
    plt.subplot(1, 2, 2)
    fig = data[var].hist(bins=20)
    fig.set_ylabel('Number of houses')
    fig.set_xlabel(var)

    plt.show()


# The majority of the continuous variables seem to contain outliers. In addition, the majority of the variables are not normally distributed. If we are planning to build linear regression, we might need to tackle these to improve the model performance. To tackle the 2 aspects together, I will do discretisation. And in particular, I will use trees to find the right buckets onto which I will divide the variables.

# #### Outliers in discrete variables
# 
# Now, let's identify outliers in numerical discrete variables. I will call outliers, those values that are present in less than 1% of the houses. This is exactly the same as finding rare labels in categorical variables. Discrete variables, in essence can be pre-processed / engineered as if they were categorical. Keep this in mind.

# In[ ]:


# outlies in discrete variables
for var in discrete:
    (data.groupby(var)[var].count() / np.float(len(data))).plot.bar()
    plt.ylabel('Percentage of observations per label')
    plt.title(var)
    plt.show()
    #print(data[var].value_counts() / np.float(len(data)))
    print()


# Most of the discrete variables show values that are shared by a tiny proportion of houses in the dataset. For linear regression, this may not be a problem, but it most likely will be for tree methods.
# 
# 
# #### Number of labels: cardinality
# 
# Let's go ahead now and examine our categorical variables. First I will determine whether they show high cardinality. This is, a high number of labels.

# In[ ]:


no_labels_ls = []
for var in categorical:
    no_labels_ls.append(len(data[var].unique()))
    
 
tmp = pd.Series(no_labels_ls)
tmp.index = pd.Series(categorical)
tmp.plot.bar(figsize=(12,8))
plt.title('Number of categories in categorical variables')
plt.xlabel('Categorical variables')
plt.ylabel('Number of different categories')


# Most of the variables, contain only a few labels. Then, we do not have to deal with high cardinality. That is good news!

# ### Separate train and test set

# In[ ]:


# Let's separate into train and test set

X_train, X_test, y_train, y_test = train_test_split(data, data.SalePrice, test_size=0.1,
                                                    random_state=0)
X_train.shape, X_test.shape


# **Now we will move on and engineer the features of this dataset. The most important part for this course.**

# ### Bespoke feature engineering
# 
# First, let's extract information from temporal variables.
# #### Temporal variables
# 
# First, we will create those temporal variables we discussed a few cells ago

# In[ ]:


# function to calculate elapsed time

def elapsed_years(df, var):
    # capture difference between year variable and year the house was sold
    df[var] = df['YrSold'] - df[var]
    return df


# In[ ]:


for var in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
    X_train = elapsed_years(X_train, var)
    X_test = elapsed_years(X_test, var)
    submission = elapsed_years(submission, var)


# In[ ]:


X_train[['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']].head()


# Instead of years, now we have the amount of years passed since the house was built or remodeled and the house was sold. Next, we drop the YrSold variable from the datasets, because we already extracted its value.

# In[ ]:


# drop YrSold
X_train.drop('YrSold', axis=1, inplace=True)
X_test.drop('YrSold', axis=1, inplace=True)
submission.drop('YrSold', axis=1, inplace=True)


# ### Engineering missing values in numerical variables 
# #### Continuous variables

# In[ ]:


# print variables with missing data
# keep in mind that now that we created those new temporal variables, we
# are going to treat them as numerical and continuous as well:

# remove YrSold because it is no longer in our dataset
year_vars.remove('YrSold')

# examine percentage of missing values
for col in numerical+year_vars:
    if X_train[col].isnull().mean()>0:
        print(col, X_train[col].isnull().mean())


# - LotFrontage and GarageYrBlt: These variables show more than 5% NA, so I create **additional variable with NA + median imputation**
# - CMasVnrArea: Less than 5% NA so: **median imputation**

# In[ ]:


# add variable indicating missingness + median imputation
for df in [X_train, X_test, submission]:
    for var in ['LotFrontage', 'GarageYrBlt']:
        df[var+'_NA'] = np.where(df[var].isnull(), 1, 0)
        df[var].fillna(X_train[var].median(), inplace=True) 

for df in [X_train, X_test, submission]:
    df['MasVnrArea'].fillna(X_train.MasVnrArea.median(), inplace=True)


# #### Discrete variables

# In[ ]:


# print variables with missing data
for col in discrete:
    if X_train[col].isnull().mean()>0:
        print(col, X_train[col].isnull().mean())


# There are no missing data in the discrete variables. Good, then we don't have to engineer them.
# 
# ### Engineering Missing Data in categorical variables 

# In[ ]:


# print variables with missing data
for col in categorical:
    if X_train[col].isnull().mean()>0:
        print(col, X_train[col].isnull().mean())


# I will add a 'Missing' Label to all of them. If the missing data are rare, I will handle those together with rare labels in a subsequent engineering step.

# In[ ]:


# add label indicating 'Missing' to categorical variables

for df in [X_train, X_test, submission]:
    for var in categorical:
        df[var].fillna('Missing', inplace=True)


# In[ ]:


# check absence of null values
for var in X_train.columns:
    if X_train[var].isnull().sum()>0:
        print(var, X_train[var].isnull().sum())


# In[ ]:


# check absence of null values
for var in X_train.columns:
    if X_test[var].isnull().sum()>0:
        print(var, X_test[var].isnull().sum())


# In[ ]:


# check absence of null values
submission_vars = []
for var in X_train.columns:
    if var!='SalePrice' and submission[var].isnull().sum()>0:
        print(var, submission[var].isnull().sum())
        submission_vars.append(var)


# There are a few variables in the submission dataset, that did not show NA in the training dataset. For these variables, and to be able to score them using machine learning algorithms, I will fill the NA with the median value.

# In[ ]:


# Fill NA with median value for those variables that show NA only in the submission set

for var in submission_vars:
    submission[var].fillna(X_train[var].median(), inplace=True)


# ### Outliers in Numerical variables (15)
# 
# In order to tackle outliers and skewed distributions at the same time, I suggested I would do discretisation. And in order to find the optimal buckets automatically, I would use decision trees to find the buckets for me.

# In[ ]:


def tree_binariser(var):
    score_ls = [] # here I will store the mse

    for tree_depth in [1,2,3,4]:
        # call the model
        tree_model = DecisionTreeRegressor(max_depth=tree_depth)

        # train the model using 3 fold cross validation
        scores = cross_val_score(tree_model, X_train[var].to_frame(), y_train, cv=3, scoring='neg_mean_squared_error')
        score_ls.append(np.mean(scores))

    # find depth with smallest mse
    depth = [1,2,3,4][np.argmin(score_ls)]
    #print(score_ls, np.argmin(score_ls), depth)

    # transform the variable using the tree
    tree_model = DecisionTreeRegressor(max_depth=depth)
    tree_model.fit(X_train[var].to_frame(), X_train.SalePrice)
    X_train[var] = tree_model.predict(X_train[var].to_frame())
    X_test[var] = tree_model.predict(X_test[var].to_frame())
    submission[var] =  tree_model.predict(submission[var].to_frame())


# In[ ]:


for var in numerical:
    tree_binariser(var)


# In[ ]:


X_train[numerical].head()


# In[ ]:


# let's explore how many different buckets we have now among our engineered continuous variables
for var in numerical:
    print(var, len(X_train[var].unique()))


# In[ ]:


for var in numerical:
    X_train.groupby(var)['SalePrice'].mean().plot.bar()
    plt.title(var)
    plt.ylabel('Mean House Price')
    plt.xlabel('Discretised continuous variable')
    plt.show()


# We can see that the mean House Price value increases with the value of the bucket. This means we managed to create a monotonic distribution between the numerical variable and the target.

# ### Engineering rare labels in categorical and discrete variables 

# In[ ]:


def rare_imputation(variable):
    # find frequent labels / discrete numbers
    temp = X_train.groupby([variable])[variable].count()/np.float(len(X_train))
    frequent_cat = [x for x in temp.loc[temp>0.03].index.values]
    
    X_train[variable] = np.where(X_train[variable].isin(frequent_cat), X_train[variable], 'Rare')
    X_test[variable] = np.where(X_test[variable].isin(frequent_cat), X_test[variable], 'Rare')
    submission[variable] = np.where(submission[variable].isin(frequent_cat), submission[variable], 'Rare')


# In[ ]:


# the following vars in the submission dataset are encoded in different types
# so first I cast them as int, like in the train set

for var in ['BsmtFullBath', 'BsmtHalfBath', 'GarageCars']:
    submission[var] = submission[var].astype('int')


# In[ ]:


# find infrequent labels in categorical variables and replace by Rare
for var in categorical:
    rare_imputation(var)
    
# find infrequent labels in categorical variables and replace by Rare
# remember that we are treating discrete variables as if they were categorical
for var in discrete:
    rare_imputation(var)


# In[ ]:


# check that we haven't created missing values in the submission dataset
for var in X_train.columns:
    if var!='SalePrice' and submission[var].isnull().sum()>0:
        print(var, submission[var].isnull().sum())
        submission_vars.append(var)


# In[ ]:


# let's check that it worked
for var in categorical:
    (X_train.groupby(var)[var].count() / np.float(len(X_train))).plot.bar()
    plt.ylabel('Percentage of observations per label')
    plt.title(var)
    plt.show()


# In[ ]:


# let's check that it worked
for var in discrete:
    (X_train.groupby(var)[var].count() / np.float(len(X_train))).plot.bar()
    plt.ylabel('Percentage of observations per label')
    plt.title(var)
    plt.show()


# Fantastic, we have replaced infrequent labels in both categorical and numerical variables. We see the presence of the label rare in both!
# 
# ### Encode categorical and discrete variables 
# 
# I will use target encoding for categorical variables. This way, the labels will be replaced by the mean of the SalePrice, and will remain in a similar scale to the one that now show our numerical variables.

# In[ ]:


def encode_categorical_variables(var, target):
        # make label to price dictionary
        ordered_labels = X_train.groupby([var])[target].mean().to_dict()
        
        # encode variables
        X_train[var] = X_train[var].map(ordered_labels)
        X_test[var] = X_test[var].map(ordered_labels)
        submission[var] = submission[var].map(ordered_labels)

# encode labels in categorical vars
for var in categorical:
    encode_categorical_variables(var, 'SalePrice')
    
# encode labels in discrete vars
for var in discrete:
    encode_categorical_variables(var, 'SalePrice')


# In[ ]:


# sanity check: let's see that we did not introduce NA by accident
for var in X_train.columns:
    if var!='SalePrice' and submission[var].isnull().sum()>0:
        print(var, submission[var].isnull().sum())


# In[ ]:


#let's inspect the dataset
X_train.head()


# We can see that the labels have now been replaced by the mean house price.
# 
# ### Feature scaling 

# In[ ]:


X_train.describe()


# We can see that because, we used the SalePrice target  to encode both our numerical continuous and discrete and categorical variables, all our variables show the mean house price as mean value. The standard deviation however, varies, following the nature of the original variable.

# In[ ]:


# let's create a list of the training variables
training_vars = [var for var in X_train.columns if var not in ['Id', 'SalePrice']]

print('total number of variables to use for training: ', len(training_vars))


# In[ ]:


training_vars


# In[ ]:


# fit scaler
scaler = MinMaxScaler() # create an instance
scaler.fit(X_train[training_vars]) #  fit  the scaler to the train set for later use


# The scaler is now ready, we can use it in a machine learning algorithm when required. See below.
# 
# ### Machine Learning algorithm building
# 
# **Note**
# 
# The distribution of SalePrice is also skewed, so I will fit the model to the log transformation of the house price.
# 
# Then, to evaluate the models, we need to convert it back to prices.
# 
# #### xgboost

# In[ ]:


xgb_model = xgb.XGBRegressor()

eval_set = [(X_test[training_vars], np.log(y_test))]
xgb_model.fit(X_train[training_vars], np.log(y_train), eval_set=eval_set, verbose=False)

pred = xgb_model.predict(X_train[training_vars])
print('xgb train mse: {}'.format(mean_squared_error(y_train, np.exp(pred))))
print('xgb train rmse: {}'.format(sqrt(mean_squared_error(y_train, np.exp(pred)))))
print()
pred = xgb_model.predict(X_test[training_vars])
print('xgb test mse: {}'.format(mean_squared_error(y_test, np.exp(pred))))
print('xgb test rmse: {}'.format(sqrt(mean_squared_error(y_test, np.exp(pred)))))


# This model shows some over-fitting. Compare the rmse for train and test.

# #### Random Forests

# In[ ]:


rf_model = RandomForestRegressor(n_estimators=800, max_depth=6)
rf_model.fit(X_train[training_vars], np.log(y_train))

pred = rf_model.predict(X_train[training_vars])
print('rf train mse: {}'.format(mean_squared_error(y_train, np.exp(pred))))
print('rf train rmse: {}'.format(sqrt(mean_squared_error(y_train, np.exp(pred)))))

print()
pred = rf_model.predict(X_test[training_vars])
print('rf test mse: {}'.format(mean_squared_error(y_test, np.exp(pred))))
print('rf test rmse: {}'.format(sqrt(mean_squared_error(y_test, np.exp(pred)))))


# This model shows some over-fitting. Compare the rmse for train and test.

# #### Support vector machine

# In[ ]:


SVR_model = SVR()
SVR_model.fit(scaler.transform(X_train[training_vars]), np.log(y_train))

pred = SVR_model.predict(X_train[training_vars])
print('SVR train mse: {}'.format(mean_squared_error(y_train, np.exp(pred))))
print('SVR train rmse: {}'.format(sqrt(mean_squared_error(y_train, np.exp(pred)))))

print()
pred = SVR_model.predict(X_test[training_vars])
print('SVR test mse: {}'.format(mean_squared_error(y_test, np.exp(pred))))
print('SVR test rmse: {}'.format(sqrt(mean_squared_error(y_test, np.exp(pred)))))


# #### Regularised linear regression

# In[ ]:


lin_model = Lasso(random_state=2909, alpha=0.005)
lin_model.fit(scaler.transform(X_train[training_vars]), np.log(y_train))

pred = lin_model.predict(scaler.transform(X_train[training_vars]))
print('Lasso Linear Model train mse: {}'.format(mean_squared_error(y_train, np.exp(pred))))
print('Lasso Linear Model train rmse: {}'.format(sqrt(mean_squared_error(y_train, np.exp(pred)))))

print()
pred = lin_model.predict(scaler.transform(X_test[training_vars]))
print('Lasso Linear Model test mse: {}'.format(mean_squared_error(y_test, np.exp(pred))))
print('Lasso Linear Model test rmse: {}'.format(sqrt(mean_squared_error(y_test, np.exp(pred)))))


# The best model is the Lasso, so I will submit only that one for Kaggle.
# 
# ### Submission to Kaggle

# In[ ]:


# make predictions for the submission dataset
final_pred = pred = lin_model.predict(scaler.transform(submission[training_vars]))


# In[ ]:


temp = pd.concat([submission.Id, pd.Series(np.exp(final_pred))], axis=1)
temp.columns = ['Id', 'SalePrice']
temp.head()


# In[ ]:


temp.to_csv('submit_housesale.csv', index=False)

