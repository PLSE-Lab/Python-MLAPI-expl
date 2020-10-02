#!/usr/bin/env python
# coding: utf-8

# # 1) Introduction
# <br>
# <div>![image.png](attachment:image.png)
# <center>Photo: https://www.amesrealestate.com/ames-real-estate/</center></div>
# 
# <p>When you read a classified add for a house, a few features are highlighted; the size of the house, the condition of the hoouse,the number of bedrooms, bathrooms, garages, the neighbourhood and special features like a pool or whether the house has more than one floor or a basement. These are the common things both the buyer and the seller highlight in most cases. The better these features are on a house, the higher the price for the house gets. 
# 
# Home buyers usually don't look for odd features like the style of masonry finish on the homeor how high the basement ceiling is.
# But what really sets the price for a house?
# 
# This competition's dataset shows that there are other influences on the price of house than the number of bedrooms or a fence. There are 79 explanatory variables given in the dataset. These variables describing, in detail, features of houses in Ames, Iowa. We must try and predict the prices of homes in Ames based on their features.</p>
# <hr>

# # 2) Importing Libraries
# First things first, we get all of those helpful libraries that we'll need to get the job done like numpy, pandas, matplotlib seaborn and   

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import numpy as np        # linear algebra
import pandas as pd       # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns     # data visualisation 

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso


import datetime     # datetime for calculating the age of things
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


from scipy import stats #for some statistics


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # 3) Getting The Data
# Lets actually begin. We assign to the variables train and test the 'test.csv' and 'train.csv' files to the variables train and test respectively.

# In[ ]:


#Data sets
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# # 4) Formatting Our Data
# We extract the ID's to avoid confusing them with variables. Its important that we keep the IDs for the test set to use for submission.  

# In[ ]:


#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']


# In[ ]:


#We drop the ID columns in both datasets
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)


# # 5) Exploratory Data Analysis
# We're going to assume that we've been given garbage data. Really filthy and dirty data. We will look for missing values and handle them, we will search for duplicates and delete the duplicates should there be any, and we we will normalise and encode some of the vaariables in the data set. We want get this data clean.
# <center>![image.png](attachment:image.png)</center>
# 

# ## 5.1) SalePrice

# ### 5.1.1) Outliers

# In[ ]:


#Outliers
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# In[ ]:


sorted_train = train.sort_values(by = ['SalePrice'])
q1,q3 = np.percentile(sorted_train['SalePrice'], [25,75])
iqr = q3-q1
lower_bound = q1 - (1.5*iqr)
upper_bound = q3 + (1.5*iqr)


# In[ ]:


q1


# In[ ]:


q3


# In[ ]:


lower_bound


# In[ ]:


upper_bound


# In[ ]:


#Removing outliers
train = train.drop(train[(train['GrLivArea']>lower_bound) & (train['SalePrice']<upper_bound)].index)

#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# ### 5.1.2) Normality/Skewness
# As an example, we will look at SalePrice and normalise it individually by first looking a histogram and qq-plot for SalePrice.

# In[ ]:


#Normality Check
sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()


# In[ ]:


#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column 
train["SalePrice"] = np.log1p(train["SalePrice"])

#Check the new distribution 
sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()


# In[ ]:


#combining tests and train into one set to avoid code repetition
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))


# ## 6) Missing Data

# ## 6.1) Missing Data Tabulated

# In[ ]:


#missing data
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)


# ## 6.2) Missing Data Visualisation

# In[ ]:


#viaual representation of the misisng data
f, ax = plt.subplots(figsize=(15, 13))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


# ## 6.3) Handling Missing Data

# In[ ]:


#missing data
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
    
#Special fills
#zeros
for col in ('GarageYrBlt','GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0) #we don't fill GarageYrBlt we create GarageAge for it.
            
all_data['GarageAge'] = datetime.datetime.now().year - all_data['GarageYrBlt'] 
all_data.drop(columns=['GarageYrBlt'], inplace = True)
            
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

#median fill
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

#mode fill
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])


#explained by data_description.txt
all_data["Functional"] = all_data["Functional"].fillna("Typ")

#drops
all_data = all_data.drop(['Utilities'], axis=1)


# ## 6.4) Last Check for Missing Data

# In[ ]:


#Check remaining missing values if any 
all_data[col] = all_data[col].fillna(0)

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()


# # 7) Multicolinearity
# When independent variables in a regression model are correlated, we call this multicolinearity. This is a problem since the independent variables should be exactly that, independent. If the degree of correlation between variables is sufficiently high, fitting a model to the data and interpreting results from the data can be problematic, because if independent variables are correlated, changes in one variable are associated with changes in another variable. We don't want this so we will remove any cases of multicolinearity.

# ### 7.1) Correlation Heatmap

# In[ ]:


#Correlation map to see how features are correlated with SalePrice
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True, cmap="YlGnBu")


# ### 7.2) Derived Variables.

# In[ ]:


all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']


# ### 7.3) Removing Correlated Predictor Variables
# We then remove the correlated predictors whose correlation with SalePrice is the lowest respectively. e.g GarageCars and GarageArea are both highly correlated, but GarageCars has has a correlation of 0.68 while GarageArea has a of 0.66 with SalePrice. So we remove GarageArea.

# In[ ]:


#New column TotalSF to keep some of the influence the soon to dropped column '2ndFlrSF'
all_data.drop(columns = ['GarageArea','1stFlrSF','TotRmsAbvGrd','2ndFlrSF'], inplace = True) 


# # 8) Variable Encoding, Normalising and Scaling.

# ## 8.1) Encoding Ordinal Categorical Variables

# In[ ]:


replace_map_landslope = {'Sev': 3, 'Mod': 2, 'Gtl': 1  }
replace_map_qual =  {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2,'Po': 1 }
replace_map_qual_na = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2,'Po': 1, 'None': 0  }
replace_map_finish = {'GLQ': 6,'ALQ': 5,'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None':0 }
replace_map_exposure = {'Gd': 4,'Av': 3,'Mn': 2, 'No': 1, 'None': 0 }
replace_map_yes_no = {'Y': 1, 'N': 0 }
replace_map_functional = { 'Typ': 8, 'Min1': 7, 'Min2':6, 'Mod':5, 'Maj1':4, 'Maj2': 3, 'Sev': 2, 'Sal':1 }
replace_map_garage_fin = { 'Fin': 3, 'RFn': 2 , 'Unf': 1, 'None': 0 }
replace_map_drive = { 'Y': 2 , 'P': 1 ,'N': 0 }
replace_map_pool = { 'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'None': 0 }
replace_map_fence = {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'None': 0 }


# This is just a function we made to encode our data. Basically does is compare the value in the column to what's in the mapping dictionary and replaces it with values consistent with the given 'description.txt' file.

# In[ ]:


def get_dict_value(key, replace_dict):
    '''Check if key is in dictionary
       if not the function must do nothing
       if the key is in the dictionary, the function must return the corresponding value'''
    if key in replace_dict:
        return replace_dict[key]
    else:
        return key


# In[ ]:


#ExterQual
all_data['ExterQual'] = all_data['ExterQual'].apply(get_dict_value, replace_dict = replace_map_qual ) 
all_data['ExterQual'] = all_data['ExterQual'].astype('int')
#ExterCond
all_data['ExterCond'] = all_data['ExterCond'].apply(get_dict_value, replace_dict = replace_map_qual ) 
all_data['ExterCond'] = all_data['ExterCond'].astype('int')
#LandSlope
all_data['LandSlope'] = all_data['LandSlope'].apply(get_dict_value, replace_dict = replace_map_landslope ) 
all_data['LandSlope'] = all_data['LandSlope'].astype('int')
#BsmtQual
all_data['BsmtQual'] = all_data['BsmtQual'].apply(get_dict_value, replace_dict = replace_map_qual_na ) 
all_data['BsmtQual'] = all_data['BsmtQual'].astype('int')
#BsmtCond
all_data['BsmtCond'] = all_data['BsmtCond'].apply(get_dict_value, replace_dict = replace_map_qual_na ) 
all_data['BsmtCond'] = all_data['BsmtCond'].astype('int')
#BsmtFinType1
all_data['BsmtFinType1'] = all_data['BsmtFinType1'].apply(get_dict_value, replace_dict = replace_map_finish ) 
all_data['BsmtFinType1'] = all_data['BsmtFinType1'].astype('int')
#BsmtFinType2
all_data['BsmtFinType2'] = all_data['BsmtFinType2'].apply(get_dict_value, replace_dict = replace_map_finish ) 
all_data['BsmtFinType2'] = all_data['BsmtFinType2'].astype('int')
#HeatingQC
all_data['HeatingQC'] = all_data['HeatingQC'].apply(get_dict_value, replace_dict = replace_map_qual ) 
all_data['HeatingQC'] = all_data['HeatingQC'].astype('int')
#CentralAir
all_data['CentralAir'] = all_data['CentralAir'].apply(get_dict_value, replace_dict = replace_map_yes_no ) 
all_data['CentralAir'] = all_data['CentralAir'].astype('int')
#KitchenQual
all_data['KitchenQual'] = all_data['KitchenQual'].apply(get_dict_value, replace_dict = replace_map_qual_na ) 
all_data['KitchenQual'] = all_data['KitchenQual'].astype('int')
#Functional
all_data['Functional'] = all_data['Functional'].apply(get_dict_value, replace_dict = replace_map_functional ) 
all_data['Functional'] = all_data['Functional'].astype('int')
#FireplaceQu
all_data['FireplaceQu'] = all_data['FireplaceQu'].apply(get_dict_value, replace_dict = replace_map_qual_na ) 
all_data['FireplaceQu'] = all_data['FireplaceQu'].astype('int')
#GarageFinish
all_data['GarageFinish'] = all_data['GarageFinish'].apply(get_dict_value, replace_dict = replace_map_garage_fin) 
all_data['GarageFinish'] = all_data['GarageFinish'].astype('int')
#GarageQual
all_data['GarageQual'] = all_data['GarageQual'].apply(get_dict_value, replace_dict = replace_map_qual_na) 
all_data['GarageQual'] = all_data['GarageQual'].astype('int')
#GarageCond
all_data['GarageCond'] = all_data['GarageCond'].apply(get_dict_value, replace_dict = replace_map_qual_na) 
all_data['GarageCond'] = all_data['GarageCond'].astype('int')


# ## 8.2) Skewness in Numeric Variables

# In[ ]:


numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head()


# ## 8.3) Normalising using the Box-Cox transformation
# We're trying to normalise all non-normal or skewed predictor variables in our data set. The Box Cox transformation is a way to do this. Normalising our data iis important or regression. The mathematical formula for a Box-Cox transformation is as follows
# <center>![image.png](attachment:image.png)</center>
# 
# ### 8.3.1) Skewness
# How would we know if our data is not normal or not? As a guide: If our data's skewness is less than -1 or greater than 1, the distribution of our data is highly skewed. If our data's skewness is between -1 and -0.5 or between 0.5 and 1, the distribution of our data is moderately skewed. If our data's skewness is between -0.5 and 0.5, the distribution is approximately symmetric. So we would like to normalise all predictor variables with a skewness value that is less than -0.5 or greater than 0.5.

# In[ ]:


skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)
    
#all_data[skewed_features] = np.log1p(all_data[skewed_features])  #In case Box cox doesn't work
#x = exp(log(lambda * transform + 1) / lambda)                    #To transform the values back


# ### 8.3.2) Nominal Categorical Variables:
# ** 8.3.2.1) Dummy Variables**
# <p>For our nominal categorical variables, we call the get dummies function from pandas. We also specify that the function drop the first column for each variable encoded in this way to avoid the dummy variable trap.</p>

# In[ ]:


all_data = pd.get_dummies(all_data)
print(all_data.shape)


# **Data Overview:**
# <p>Lets have a complete look at our data after all that effort.</p>

# In[ ]:


all_data.info


# In[ ]:


train = all_data[:ntrain]
test = all_data[ntrain:]


# ### 8.3.3) One last check for missing data

# In[ ]:



#missing data
test_na = (test.isnull().sum() / len(test)) * 100
test_na = test_na.drop(test_na[test_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame( {'Missing Ratio' :test_na} )
missing_data.head(20)


# In[ ]:


test['GarageAge'] = test['GarageAge'].fillna(0)


# ### 8.3.4) Scaling our data
# But why are we scaling again? Since we added dummy variables, we must scale them relative to the data.
# 
# Most machine learning algorithms only take the size of the predictors and not the actual units those predictors were recorded in. Variables recorded in millimeters, for example, would have a greater influence on the model than values recorded in meters (1m = 1000mm)
# 
# For this reason we will scale our data to avoid negatively impacting our models. We will use RobustScaler to do this. RobustScaler will also minimise the effects of outliers we did not remove from the predictor variables.

# In[ ]:


sc=RobustScaler()
x=sc.fit_transform(train)
test=sc.transform(test)


# In[ ]:


test


# # 9)Train-Test-Split

# In[ ]:


X_train_lasso, X_test_lasso, y_train_lasso, y_test_lasso = train_test_split(x,
                                                    y_train,
                                                    test_size=0.201,
                                                    shuffle=False)


# # 10) Regression Models

# ## 10.1) Lasso (least absolute shrinkage and selection operator) Regression

# ### 10.1.1) Overfit Lasso Model
# We were uncertain of whether or not we should fit a model to the given data without having or own train/test split so we submitted this model to competition as a test to see if we would get a better score or not.

# In[ ]:


def lasso_model():
    '''
    This function fits the train data given by the competition and cleaned, to a Lasso Regression model
    '''
    lasso = Lasso(alpha =0.001, random_state=1)
    model = lasso.fit(x,y_train)
    return model
pred = lasso_model().predict(test)
preds = np.exp(pred)


# ### 10.1.2) Lasso Model 2
# We fit a model to our own train/test split in order to compare errors and r-squared values.

# In[ ]:


def Lasso_model_2():
    '''
    This function fits the train/split data to a Lasso Regression model
    '''
    train_test_lasso = Lasso(alpha=0.0001, random_state=1)
    model = train_test_lasso.fit( X_train_lasso, y_train_lasso )
    return model
pred_train_lasso = Lasso_model_2().predict( X_train_lasso )
pred_test_lasso = Lasso_model_2().predict( X_test_lasso )


# ## 10.2) LinearRegression
# We also built a LinearRegression model to compare to the other regression techniques we used for this competition.

# In[ ]:


from sklearn.linear_model import LinearRegression
def LM_model():
    '''
    This function fits the data to a Linear Regression
    '''
    lm = LinearRegression()
    model = lm.fit(X_train_lasso, y_train_lasso)
    return model

pred_train_lm = LM_model().predict( X_train_lasso )
pred_test_lm = LM_model().predict( X_test_lasso )


# In[ ]:


print(LM_model().coef_)


# ## 10.3) Ridge Regression
# We also built a LinearRegression model to compare to the other regression techniques we used for this competition.

# In[ ]:


from sklearn.linear_model import Ridge
def Ridge_model():
    '''
    This function fits the data to a Ridge Regression
    '''
    ridge = Ridge()
    model = ridge.fit(X_train_lasso, y_train_lasso)
    return model
pred_train_ridge = Ridge_model().predict( X_train_lasso )
pred_test_ridge = Ridge_model().predict( X_test_lasso )


# ## 10.4) Functions built to make camparison easier
# We built a few functions to make returning stats easier for us.

# In[ ]:


#function to get rmse, r-squuared and adjusted r-squared as long as the training set is passed to the function
def model_acc_stats( this_x_, actual, predicted):
    '''
    This function receives the X-values, the actual y values and the predicted y values for the model.
    '''
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 =  r2_score(actual, predicted)
    adj_r2 = 1 - (1- r2)* ((this_x_.shape[0] - 1)/(this_x_.shape[0] - this_x_.shape[1] - 1))
    return {'RMSE': rmse, 'R-squared': r2, 'Adj_R-squared': adj_r2 }


# In[ ]:


def model_acc(this_x, model_title,actual_y_train, predicted_y_train, actual_y_test, predicted_y_test, get):
    '''
    This function receives the X-values, the model title,  the actual y values for the training set,
    the predicted y values for the training set, actual y values for the testing set,
    and the predicted values for the testing set.
    '''
    train_stats = model_acc_stats( this_x, actual_y_train, predicted_y_train )
    test_stats = model_acc_stats( this_x, actual_y_test, predicted_y_test )
    
    if get ==  True:
        return [ train_rmse, test_rmse, train_r2, test_r2, train_adj_r2, test_adj_r2 ]
    else:
        print( model_title )
        print( "(Training, Test)" )
        print( "RMSE: (%f,%f)" % (train_stats['RMSE'], test_stats['RMSE']))
        print( "R-squared: (%f,%f)" % (train_stats['R-squared'], test_stats['R-squared']))
        print( "Adjusted R-squared: (%f,%f)" % (train_stats['Adj_R-squared'], test_stats['Adj_R-squared'] ))


# ## 10.5) Results
# We run our function and get the RMSE, R-squared and Adjusted R-squared for our models and we conclude on which regression technique and model to use.

# ## 10.5.1) Runnnig our functions

# In[ ]:


#get rsme, r-squared and adjusted r-squatred for each model
model_acc( x,"Linear Regression",y_train_lasso, pred_train_lm, y_test_lasso, pred_test_lm, False )


# In[ ]:


#get rsme, r-squared and adjusted r-squatred for each model
model_acc( x,"Ridge Regression",y_train_lasso, pred_train_ridge, y_test_lasso, pred_test_ridge, False )


# In[ ]:


#get rsme, r-squared and adjusted r-squatred for each model
model_acc( x,"Lasso Regression",y_train_lasso, pred_train_lasso, y_test_lasso, pred_test_lasso, False )


# ## 10.5.2) Conclusion
# We found that Ridge and Lasso Regression work best for this case. Lasso Regression was marginally better than the Ridge model since the Rsquared values for the Lasso Model were higher in both training and testing sets, and the RMSE was lower for both training and testing. They are almost identical in their predictions.
# 
# The Linear Regression Model overfit the data and that is seen by the abnormally low R-squared values for the training set and the low RMSE for the training set but a ridiculously high value for the testing set.

# ## 10.5.3) Getting the Coefficients for Interpretation
# We get the coefficients of the Lasso model to determine which fetures have the highest level of importance to the model. That is, what greatly affects SalePrice.

# In[ ]:


# Get the coefficients
lasso_coef = Lasso_model_2().coef_
print(lasso_coef)

# Plot the coefficients
f, ax = plt.subplots(figsize=(100, 5))
plt.xticks(range(len(train.columns)), train.columns.values, rotation=90, size=10)
plt.margins(0,0)
sns.lineplot(x=range(len(train.columns)), y=lasso_coef, sort=False, lw=1)
plt.show()


# In[ ]:


#getting the values instead to see the results better
coef_df = pd.DataFrame( columns = ['Variable','Coef'])
coef_df['Variable'] = train.columns.values
coef_df['Coef'] = lasso_coef # to see the variables magnitude and direction(positive or negative)
coef_df['AbsCoef'] = np.abs(lasso_coef) # to see just the variable's magnitude


# In[ ]:


#feature selection as done by the lasso regression technique
df_selected_feats = coef_df[(coef_df['AbsCoef'] != 0) & (coef_df['AbsCoef'] > 0.001)].sort_values(by=['AbsCoef'] , ascending= False)
features_kept = df_selected_feats['Variable'].values


# In[ ]:


#The features with the most impact
df_selected_feats.sort_values('AbsCoef', ascending=False).head()


# In[ ]:


#The features impact with their weights
coef_df.sort_values('Coef', ascending=False).head(10)


# In[ ]:


all_data_2 = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')


# In[ ]:


all_data_2.sort_values('SalePrice', ascending=False).head(10)


# In[ ]:


coef_df.sort_values('Coef', ascending=False).tail(10)


# In[ ]:


all_data_2.sort_values('SalePrice', ascending=False).head(10)


# A use case for the coeficients is to compare the most expensive houses and see how many of the features with positive coefficients the houses have and how many of the feutures with negative coefficients the expensive houses don't have.
# For example: it seems that houses with a stone foundation, a high total surface area and wood shingles for roofing and in the StoneBr and or  

# In[ ]:


#These are the features we could apply to the LinearRegression model to see if we can't get a better model for future reference
features_kept


# ## 10.5.4) Residual Plots:
# We plot the difference between the actual data and the predicted values to see how our models performed.

# In[ ]:


#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(y_train, Lasso_model_2().predict(x))
plt.title('Lasso Regression: Actual vs Predicted')
plt.ylabel('Predicted', fontsize=13)
plt.xlabel('Actual', fontsize=13)
plt.show()


# In[ ]:


#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(y_train, Ridge_model().predict(x))
plt.title('Ridge Regression: Actual vs Predicted')
plt.ylabel('Predicted', fontsize=13)
plt.xlabel('Actual', fontsize=13)
plt.show()


# In[ ]:


#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(y_train, LM_model().predict(x))
plt.title('Linear Regression: Actual vs Predicted')
plt.ylabel('Predicted', fontsize=13)
plt.xlabel('Actual', fontsize=13)
plt.show()


# # 11) Final Submission
# <center>
# ![image.png](attachment:image.png)
# Image: https://cognigen-cellular.com/explore/mail-clipart-sent-mail/
# </center>

# In[ ]:


ridge_output=pd.DataFrame({'Id':test_ID.values, 'SalePrice':np.exp(Ridge_model().predict(test))})
linear_output=pd.DataFrame({'Id':test_ID.values, 'SalePrice':np.exp(LM_model().predict(test))})
lasso_1_output=pd.DataFrame({'Id':test_ID.values, 'SalePrice':np.exp(lasso_model().predict(test))})
lasso_2_output=pd.DataFrame({'Id':test_ID.values, 'SalePrice':np.exp(Lasso_model_2().predict(test))})


# In[ ]:


ridge_output.head()


# In[ ]:


linear_output.head()


# In[ ]:


lasso_1_output.head()


# In[ ]:


lasso_2_output.head()


# In[ ]:


ridge_output.to_csv('ridge_submission.csv', index=False)
lasso_1_output.to_csv('lasso_1_submission.csv', index=False)
lasso_2_output.to_csv('lasso_2_submission.csv', index=False)
linear_output.to_csv('linear_submission.csv', index=False)


# <center>
#     <image style= 'height: 50%; width: 50%;'>![image.png](attachment:image.png)</image>
#     Image:http://clipart-library.com/bowing-cliparts.html
# </center>
# <center >## Thank you for your time and we hope you found this notebook informative and helpful.</center>

# In[ ]:




