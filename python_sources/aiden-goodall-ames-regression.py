#!/usr/bin/env python
# coding: utf-8

# # <center> Ames Housing Project </center>

# # <center> Regression Challenge Documentation </center>
# <a id="index"></a>

# ___________________________________________________________________________

# # Table of contents

# 1. [Introduction](#1)
# 2. [Importing the essential libraries and data](#2)
# 3. [First data inspection](#3)
# 4. [Data cleaning on training data](#4)
# 5. [Removing outliers](#5)
# 6. [Data exploration](#6)
# 7. [Data cleaning testing data](#7)
# 8. [Checking the dataframe shapes](#8)
# 9. [Data transformation](#9)
# 10. [Model fitting and exploration](#10)
#     * [Lasso](#lasso)
#     * [Ridge](#ridge)
#     * [ElasticNet](#elastic)
#     
#     
# 11. [Testing the model](#11)
# 13. [Still to do](#12)
# 14. [Acknowledgements](#13)

# <a id="1"></a>
# ___________________________________________________________________________

# # 1. Introduction

# ### Competition introduction on kaggle
# Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.
# 
# With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

# ### My introduction
# I was asked to work with the provided dataset and build a regression model, that has the ability to predict what a specific house will be worth based on its attributes. In this document I will explain the choices I made in going about creating my model.

# [Return to index](#index)

# <a id="2"></a>
# ___________________________________________________________________________

# # 2. Importing the essential libraries and data 

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train_df = pd.read_csv('../input/train.csv', index_col='Id')


# In[ ]:


test_df = pd.read_csv('../input/test.csv', index_col='Id')
submissions = test_df.index
test_y = pd.read_csv('../input/sample_submission.csv')


# [Return to index](#index)

# <a id="3"></a>
# _________________________________________________________________________________________________

# 
# # 3. First data inspection

# Here I have a quick look at the data saving the amount of rows currently in the dataframe to use for validation after i finish with the feature engineering and data cleaning.
# 
# afterwards exploring what are the data types for each column while exploring the data description document provided, I changed some of the data types to better fit the appropriate data types as I perceived it. 

# In[ ]:


unclean_train_shape = train_df.shape[0]
unclean_train_shape


# In[ ]:


unclean_test_shape = test_df.shape[0]
unclean_test_shape


# In[ ]:


#Using a list generator to seperate the qualitive "catagorical" and numerical "numerical" columns
numerical = [f for f in train_df.columns if train_df.dtypes[f] != 'object']
numerical.remove('SalePrice')
catagorical = [f for f in train_df.columns if train_df.dtypes[f] == 'object']


# In[ ]:


print(f'list length:{len(numerical)}\n\n{numerical}')
print("")
print(f'lest length:{len(catagorical)}\n\n{catagorical}')


# In[ ]:


# Changing numerical data to catagorical data 
catagorical_numbers = ['MSSubClass', 'OverallQual', 'OverallCond', 'YearBuilt',
                       'YearRemodAdd', 'GarageYrBlt', 'MoSold', 'YrSold']

for catagorical_test in catagorical_numbers:
    if catagorical_test in numerical:
        print(catagorical_test)
        numerical.remove(catagorical_test)
        catagorical.append(catagorical_test) 


# In[ ]:


# Changing numerical data to catagorical data 
train_df[catagorical_numbers] = train_df[catagorical_numbers].astype(str)


# In[ ]:


# Changing numerical data to catagorical data 
test_df[catagorical_numbers] = test_df[catagorical_numbers].astype(str)


# In[ ]:


print(f'list length:{len(numerical)}\n\n{numerical}')
print('')
print(f'list length:{len(catagorical)}\n\n{catagorical}')


# [Return to index](#index)

# <a id="4"></a>
# _________________________________________________________________________________________________

# 
# #  4. Data Cleaning

# In[ ]:


def percent_missing(df):
    """
    Function checks the percentage of the missing data
    within every column inside a dataframe.
    
    args:
    df: The dataframe you want to explore missing data in.
    
    returns: 
    A dictionary of every column with missing data as the key
    and the percentage of data missing within the column as the value
    """
    
    percent_missing = {}
    nulls = df.isnull().sum()
    for column in df:
        percent_missing[column] = round((nulls[column] / len(df[column])) * 100, 2)
    
    percentage_missing = {key:val for key, val in percent_missing.items() if val != 0}
    
    return percentage_missing


# I decided to treat the training and testing data differently in this challenge, with the goal of making the minimalizing assumptions made within the training data. 
# 
# I decided if there is less than 1% of the data missing in the column I will just drop the rows of missing data. 
# 
# I also explored the data description document and many of the columns with missing values has "NA" as a categorical feature, which pandas transforms to missing data. So for those columns I made the assumption all the missing values were meant to be a the categorical variable referred to as "NA" instead of missing data.
# 
# I also consulted a pair of real estate agents who informed me that as far as they were aware the law does not allow for you to have no lot frontage. I decided to fill the missing data there with the median per neighborhood since I had not looked for or removed any outliers yet.

# In[ ]:


train_df.shape


# In[ ]:


train_df.head()


# In[ ]:


percent_missing(train_df)


# In[ ]:


#All of these columns had less than 1% of their data missing I found it best to drop those rows.
train_df = train_df.dropna(subset=['Electrical', 'MasVnrType', 'MasVnrArea'])


# In[ ]:


#All these variables were related basement data so I transformed the missing data in one go
train_df[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1','BsmtFinType2']] = train_df[['BsmtQual', 'BsmtCond', 
                                                                                              'BsmtExposure', 'BsmtFinType1',
                                                                                              'BsmtFinType2']].fillna('none', axis=0)
#All these variables were related garage data so I transformed the missing data in one go
train_df[['GarageType','GarageFinish','GarageQual','GarageCond']] = train_df[['GarageType','GarageFinish',
                                                                              'GarageQual','GarageCond']].fillna('none', axis=0)

#None of these variables are related so I just did them individually 
train_df['Alley'] = train_df["Alley"].fillna('none', axis=0)
train_df['FireplaceQu'] = train_df['FireplaceQu'].fillna('none', axis=0)
train_df['PoolQC'] = train_df['PoolQC'].fillna('none', axis=0)
train_df['Fence'] = train_df['Fence'].fillna('none', axis=0)
train_df['MiscFeature'] = train_df['MiscFeature'].fillna('none', axis=0)


# In[ ]:


train_df['LotFrontage'] = train_df.groupby('Neighborhood')['LotFrontage'].transform(lambda lot_front: lot_front.fillna(lot_front.median()))


# In[ ]:


#Testing to see if there was any missing data I missed
percent_missing(train_df)


# [Return to index](#index)

# <a id="5"></a>
# _________________________________________________________________________________________________

# 
# # 5. Removing outliers

# In my investigation of the data I used histograms to visualize the data and look for outliers. Instead of spending a large amount of time writing code to plot every useful numerical column I wanted to explore for outliers 
# I used a looping function to plot every numerical column. 
# 
# I then removed the data that was significantly separated and contained a small amount of data in that separation. I ran the plotting loop again to see what the data looked after I removed the rows.
# 
# I do realize that visually judging if something is an outlier or not is not an appropriate method and I will explore using a more statistically acceptable method in future

# In[ ]:


#Plotting a histogram each numerical to locate potential outliers idea for the mass plotting code was from Jason Adams
for chart in train_df[numerical]:
    f, ax = plt.subplots(figsize=(10,6))
    train_df[chart].hist(bins=30)
    plt.title(chart)


# In[ ]:


#testing the shape of the training data before removing percieved outliers
train_df.shape


# In[ ]:


#I removed all the rows relating to values I percieved as outliers
train_df = train_df[train_df['LotFrontage'] < 200]
train_df = train_df[train_df['LotArea'] < 100000]
train_df = train_df[train_df['MasVnrArea'] < 1200]
train_df = train_df[train_df['BsmtFinSF1'] < 3000]
train_df = train_df[train_df['TotalBsmtSF'] < 3000]
train_df = train_df[train_df['1stFlrSF'] < 3000]
train_df = train_df[train_df['GrLivArea'] < 4000]
train_df = train_df[train_df['EnclosedPorch'] < 400]
train_train_df = train_df[train_df['OpenPorchSF'] < 400]
train_df.shape


# In[ ]:


#Plotting a histogram each numerical to visualize the diffrence without percieved outliers 
for chart in train_df[numerical]:
    f, ax = plt.subplots(figsize=(10,6))
    train_df[chart].hist(bins=30)
    plt.title(chart)


# [Return to index](#index)

# <a id="6"></a>
# _________________________________________________________________________________________________

# # 6. Data exploration

# After removing all the missing data and perceived outliers I then decided to explore the data slightly looking for multi-collinearity.
# 
# To do this I plotted a heat map of the of every columns correlation with each other and removed one of the two columns if they were over +/- 0.65 correlated. Then I removed the column that had the lesser correlation with the dependent variable.
# 
# I plotted the heat map a second time to see if I missed anything
# 
# Why the number +/- 0.65? I have no good exact statistical reason why I picked this number. The true reason was I felt that the majority of the correlated columns were under the selected number.
# 
# I realize that I should do deeper exploration on the selection of the correlation threshold and will do so in future.

# In[ ]:


#Looking for the sum of missing data in the training dataframe
print(train_df.isnull().sum().sum())


# In[ ]:


train_df.head()


# In[ ]:


#Plotting a heatmap of corrolations to remove multi-collinearity

plt.rc('xtick', labelsize=25) 
plt.rc('ytick', labelsize=25) 
plt.figure(figsize=[35,25])
sns.heatmap(train_df.corr(), annot=True)


# In[ ]:


#Removing the corrolated columns in the training data
train_df = train_df.drop(['GarageArea', '1stFlrSF', 'TotRmsAbvGrd', '2ndFlrSF', 'BedroomAbvGr', 'BsmtFullBath'], axis=1)


# In[ ]:


#Removing the corrolated columns in the testing data
test_df = test_df.drop(['GarageArea', '1stFlrSF', 'TotRmsAbvGrd', '2ndFlrSF', 'BedroomAbvGr', 'BsmtFullBath'], axis=1)


# In[ ]:


plt.rc('xtick', labelsize=25) 
plt.rc('ytick', labelsize=25) 
plt.figure(figsize=[35,25])
sns.heatmap(train_df.corr(), annot=True)


# [Return to index](#index)

# <a id="7"></a>
# _________________________________________________________________________________________________

# # 7. Data cleaning testing data

# With the testing data I had to make assumptions on all the missing data because for the kaggle competition every single row has to be tested otherwise I treated the testing data the same as the training. 
# 
# I decided to fill the missing numerical values with 0 because they were all columns that had capability of having 0 In the data. 
# 
# As for the categorical variables I filled them with the mode of each column. I Tried to do this automatically but after hours of continuous errors I gave up and assigned the variables manually

# In[ ]:


test_df.shape


# In[ ]:


test_df.head()


# In[ ]:


percent_missing(test_df)


# In[ ]:


#All these variables were related basement data so I transformed the missing data in one go
test_df[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1','BsmtFinType2']] = test_df[['BsmtQual', 'BsmtCond', 
                                                                                            'BsmtExposure', 'BsmtFinType1',
                                                                                            'BsmtFinType2']].fillna('none', axis=0)

#All these variables were related garage data so I transformed the missing data in one go
test_df[['GarageType','GarageFinish','GarageQual','GarageCond']] = test_df[['GarageType','GarageFinish',
                                                                            'GarageQual','GarageCond']].fillna('none', axis=0)

#None of these variables are related so I just did them individually 
test_df['Alley'] = test_df["Alley"].fillna('none', axis=0)
test_df['FireplaceQu'] = test_df['FireplaceQu'].fillna('none', axis=0)
test_df['PoolQC'] = test_df['PoolQC'].fillna('none', axis=0)
test_df['Fence'] = test_df['Fence'].fillna('none', axis=0)
test_df['MiscFeature'] = test_df['MiscFeature'].fillna('none', axis=0)
test_df['LotFrontage'] = test_df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

'''
The reason I didn't just transform all the data using the catagorical list I made earlier 
was because not all the catagorical columns use NA as a discriptor 
'''


# In[ ]:


test_df['GarageCars'] = test_df['GarageCars'].fillna(0, axis=0)

test_df['MasVnrArea'] = test_df['MasVnrArea'].fillna(0, axis=0)

test_df[['TotalBsmtSF','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF', 'BsmtHalfBath']] = test_df[['TotalBsmtSF','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF', 'BsmtHalfBath']].fillna(0, axis=0)


# In[ ]:


percent_missing(test_df)


# In[ ]:


#Finding the mode manually because attempts to automatically do it all failed
modes = ['MSZoning','Utilities','Exterior1st','Exterior2nd','MasVnrType','KitchenQual','Functional','SaleType']
for i in modes:
    print(test_df[i].value_counts())


# In[ ]:


#Manually filling na values with mode in hindsight I should've filled the mode per neighbourhood
test_df['MSZoning'] = test_df['MSZoning'].fillna('RH')
test_df['Utilities'] = test_df['Utilities'].fillna('AllPub')
test_df['Exterior1st'] = test_df['Exterior1st'].fillna('VinylSd')
test_df['Exterior2nd'] = test_df['Exterior2nd'].fillna('510')
test_df['MasVnrType'] = test_df['MasVnrType'].fillna('None')
test_df['KitchenQual'] = test_df['KitchenQual'].fillna('757')
test_df['Functional'] = test_df['Functional'].fillna('Typ')
test_df['SaleType'] = test_df['SaleType'].fillna('WD')


# In[ ]:


percent_missing(test_df)


# [Return to index](#index)

# <a id="8"></a>
# _________________________________________________________________________________________________

# # 8. Checking the dataframe shapes

# I quickly just looked if I removed all the missing data and the number of rows in my dataframe before transforming the data.

# In[ ]:


print(train_df.isnull().sum().sum())
print(test_df.isnull().sum().sum())


# In[ ]:


clean_train_shape = train_df.shape[0]


# In[ ]:


clean_test_shape = test_df.shape[0]


# In[ ]:


print(f'{unclean_train_shape - clean_train_shape}    : Training rows dropped\n{unclean_train_shape}  : Total rows in training\n{round(((unclean_train_shape - clean_train_shape) / unclean_train_shape) * 100, 2)}% : Of rows dropped from training\n\nNo rows were dropped from the testing')


# [Return to index](#index)

# <a id="9"></a>
# _________________________________________________________________________________________________

# # 9. Data transformation

# After research and exploration of other models I found that regression models work better with normalized and uniformly distributed data. 
# 
# I tried using a natural logarithm, a 1 base logarithm and a gamma distribution for the sale data finding that the natural log worked best with my model. I did test the resulting distributions using a bell curve formula but neglected to include them I will add the test in future iterations.
# 
# I cannot confidently say exactly why as I don't have the statistical or mathematical background to give an honest answer. I suspect that having the data normalized helps the model to better understand how data on different scales effect each other .

# In[ ]:


plt.rc('xtick', labelsize=10) 
plt.rc('ytick', labelsize=10) 
train_df["SalePrice"].hist(bins=30)


# In[ ]:


train_df["SalePrice"] = np.log(train_df["SalePrice"])


# In[ ]:


train_df["SalePrice"].hist(bins=30)


# In[ ]:


plt.rc('xtick', labelsize=10) 
plt.rc('ytick', labelsize=10) 

test_y["SalePrice"].hist(bins=30)


# In[ ]:


test_y["SalePrice"] = np.log(test_y["SalePrice"])


# In[ ]:


plt.rc('xtick', labelsize=10) 
plt.rc('ytick', labelsize=10) 
test_y["SalePrice"].hist(bins=30)


# Here I import skew, boxcox1p, boxcox_normmax reason being is I want to get as normal a distribution as possible with my numerical data. the reason being that if your data is skewed it has the potential to confuse your model and may lead to bias.
# 
# the reason I chose these library are;
# 
# skew: will look at the distribution of the numerical data and return the skewness of the data, the closer to 0 the data the less skew it is.
# 
# boxcox: will view the data and select the best mathematical function for this column.
# 
# boxcox_normmax: will transform the data with the best suited mathematical function in an attempt to normalize it as best as possible.
# 
# A large portion of my decision to do this is inspired from blog post by 
# [minitab](https://blog.minitab.com/blog/applying-statistics-in-quality-projects/how-could-you-benefit-from-a-box-cox-transformation).

# In[ ]:


from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax


# In[ ]:


def normalize(df):
    """
    Function looks at every numerical column and tests how skew it is,
    if the results is over 0.5 it will apply 
    box-cox and box-cox transfromation to that column of data.
    
    args:
    df: The dataframe you want to explore missing data in.
    
    returns: 
    the columns after they have been normilized
    """
    is_number = []
    for i in df.columns:
        if df[i].dtype in [ 'float16', 'int16', 'float32', 'int32', 'float64', 'int64']:
            is_number.append(i)
    skew_df = df[is_number].apply(lambda x: skew(x)).sort_values(ascending=False)

    high_skew = skew_df[skew_df > 0.5]
    skew_index = high_skew.index

    for is_skew in skew_index:
        df[is_skew] = boxcox1p(df[is_skew], boxcox_normmax(df[is_skew] + 1))


# In[ ]:


normalize(train_df)


# In[ ]:


normalize(test_df)


# Splitting the dependent variable from the training dataframe and creating the dependent for the testing dataframe.

# In[ ]:


X = train_df.drop(["SalePrice"], axis=1)
y = train_df["SalePrice"]
y_test = test_y['SalePrice']
print(f"      X:{X.shape}\ntest_df:{test_df.shape}\n      y:{y.shape}\n y_test:{y_test.shape}")


# I wanted to try make it so the training and testing dataframes never touch but unfortunately they have different categories causing different amounts of columns after creating dummy variables.
# 
# To combat data leakage what I do is write a quick print statement that will return true if the amount of rows were exactly the same after I split them back into training and testing.

# In[ ]:


dummies = pd.concat([X, test_df], sort=False)
dummies = pd.get_dummies(dummies, drop_first=True)
dummies.shape


# In[ ]:


dummies.head()


# In[ ]:


X = dummies[:1436]
X_test = dummies[1436:] 
print(f'    X is equal to train_df size : {X.shape[0] == len(train_df)} {X.shape}\nX_test is equal to test_df size : {X_test.shape[0] == len(test_df)} {X_test.shape}')


# Since I was going to use regression models that require scaled data I explored how to scale the data.
# 
# originally I tried using a standard scaler which standardized features by removing the mean and scaling it to the unit variance. The problem however is standard scaler is not good when there are outliers in the data and it appears as though I did not properly remove enough outliers in the data
# 
# I then moved on to a robust scaler because it is better suited to scaling data with many outliers by removing the median and scaling the data according to quantile ranges

# In[ ]:


from sklearn.preprocessing import RobustScaler
rs = RobustScaler()
#ss = StandardScaler()


# In[ ]:


print(X_test.isnull().sum().sum())


# In[ ]:


X = rs.fit_transform(X)
x_test = rs.fit_transform(X_test)


# [Return to index](#index)

# <a id="10"></a>
# _________________________________________________________________________________________________

# # 10. Model preparation and selection

# I used Kfolds because it generally results in a less biased model by, splitting the training data into n sections and testing each individual section before finalizing the model.
# 
# I chose 10 as the number of splits because I wanted roughly 10% of the data to be tested in the operation so dividing it by 10 seemed to be the best method of achieving that.
# 
# My understanding was achieved by reading the documentation and from a Blog post by [machinelearningmastery](https://machinelearningmastery.com/k-fold-cross-validation/)
# 
# I attempted to use Grid-Search to select the optimal hyper-parameters but never managed to get it working. 
# 
# I intend to eventually try a stacked regression which is why I imported make pipeline.
# 
# My selected models didn't have a significant reason for selection it was mainly due to taking a shotgun approach and trying multiple models for the best result.
# 
# I also left the default hyper-parameters In my current work.

# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn import metrics


# In[ ]:


get_ipython().run_line_magic('pinfo', 'KFold')


# In[ ]:


kf = KFold(n_splits=10, shuffle=True, random_state=19)


# [Return to index](#index)

# <a id="lasso"></a>
# _________________________________________________________________________________________________

# # Lasso

# Lasso stands for least absolute selection and shrinkage operator.
# 
# What it basically does is tries reduce the coefficients for the variables, The higher the alpha the closer the variables coefficients are to zero.
# 
# The shrinkage allows for the variables most strongly associated with the depended to stand out.

# In[ ]:


from sklearn.linear_model import LassoCV


# In[ ]:


lasso = LassoCV()


# In[ ]:


lasso.fit(X,y)


# In[ ]:


f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15,5))
ax1.plot(y[:50], c='g')
ax1.plot(lasso.predict(X[:50]), c='c')

ax2.plot(y_test[:50], c='g')
ax2.plot(lasso.predict(x_test[:50]), c='c');


# In[ ]:


print(f'train error: {metrics.mean_squared_error(y, lasso.predict(X))}\ntest error: {metrics.mean_squared_error(y_test, lasso.predict(x_test))}')


# [Return to index](#index)

# <a id="ridge"></a>
# _________________________________________________________________________________________________

# # Ridge

# In short and simple terms Ridge received its name because if the data has multicollinearity the model will return a ridge when plotted in 3d.
# 
# What it basically does is it applies a punishment to variables with coefficients to far away from zero.
# 
# Which should help minimizing the sum of squared residuals.

# In[ ]:


from sklearn.linear_model import RidgeCV


# In[ ]:


ridge = RidgeCV()
ridge.fit(X, y)


# In[ ]:


f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15,5))
ax1.plot(y[:50], c='g')
ax1.plot(ridge.predict(X[:50]), c='c')

ax2.plot(y_test[:50], c='g')
ax2.plot(ridge.predict(x_test[:50]), c='c');


# In[ ]:


print(f'train error: {metrics.mean_squared_error(y, ridge.predict(X))}\ntest error: {metrics.mean_squared_error(y_test, ridge.predict(x_test))}')


# [Return to index](#index)

# <a id="elastic"></a>
# _________________________________________________________________________________________________

# #  ElasticNet

# ElasticNet was created because lasso was to depended and the data which made it unstable.
# 
# What ElasticNet does is combines the penalties of lasso and ridge regression to make a more robust less unstable model

# In[ ]:


from sklearn.linear_model import ElasticNetCV


# In[ ]:


elastic = ElasticNetCV()
elastic.fit(X,y)


# In[ ]:


f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15,5))
ax1.plot(y[:50], c='g')
ax1.plot(elastic.predict(X[:50]), c='c')

ax2.plot(y_test[:50], c='g')
ax2.plot(elastic.predict(x_test[:50]), c='c');


# In[ ]:


print(f'train error: {metrics.mean_squared_error(y, elastic.predict(X))}\ntest error: {metrics.mean_squared_error(y_test, elastic.predict(x_test))}')


# [Return to index](#index)

# <a id="11"></a>
# _________________________________________________________________________________________________

# # 11. Testing the model

# Creating a csv that can be uploaded to kaggle and test my models predictive accuracy using the decided upon evaluation method.

# In[ ]:


#Submit
submission = pd.DataFrame()
submission['Id'] = submissions
feats = test_df.select_dtypes(
       include=[np.number]).interpolate()
predictions = lasso.predict(x_test)
final_predictions = np.exp(predictions)
submission['SalePrice'] = final_predictions
submission.to_csv('lasso submission.csv', index=False)


# [Return to index](#index)

# <a id="12"></a>
# _________________________________________________________________________________________________

# # 12. Still do list 

# * Explore more statistically significant methods of detecting outliers.
# * Explore different selections of the correlation thresholds.
# * Add my statistical testing used to decided on what normalization method
# * Select proper hyper-parameters in my models.
#     * get grid-search to work so I can select the optimal hyper-parameters.
# * Explore why certain models performed better than others.
#     * I don't have the statistical knowledge to confidently say why X model performed better.
# * Attempt more models of varying and explore different Ensemble methods.

# [Return to index](#index)

# <a id="13"></a>
# _________________________________________________________________________________________________

# # $$Acknowledgement$$ 

# ##### A large part of my work was my own but I got inspirations and guidance from a few sources

# 1. **Bryan Davies** and **Nicholas Meyers** helped me fix issues in my code
# 2. **Nicholas Meyers** encouraged me to explore cross-validation, Kfolds and grid-search
# 3. **Jason Adams** gave me the idea I used for plotting the histograms to look for outliers was provided by 
# 4. **Samuealleeuw** provided the code needed to make the csv for the kaggle submission.
# 5. **Shezz Basha** helped in making my notebook as clean as possible
# 6. **Dean De Cock** compiled and proved the [Dataset](http://jse.amstat.org/v19n3/decock.pdf)  
# 5. My current understanding of the models I used came from [Datacamp](https://www.datacamp.com/community/tutorials/tutorial-ridge-lasso-elastic-net)
# 7. Guidance to how I managed some of my data was from different kaggle discussions 
#     1. [Stacking for beginners](https://www.kaggle.com/niteshx2/top-50-beginners-stacking-lgb-xgb)
#     2. [Regression: top 20%](https://www.kaggle.com/goldens/regression-top-20-with-a-very-simple-model-lasso)

# [Return to index](#index)
