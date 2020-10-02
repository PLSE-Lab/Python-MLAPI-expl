#!/usr/bin/env python
# coding: utf-8

# # House Price Prediction

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import xticks

import warnings
warnings.filterwarnings('ignore')

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
pd.pandas.set_option('display.max_columns', None)


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn import metrics


# # Data Analysis & EDA

# In[ ]:


# load the dataset
house = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
house.head()


# In[ ]:


house.shape


# ######  Wohooo!!! 81 features!! This dataset would surely require lots of data analysis!!!

# In[ ]:


# check the dataset
house.info()


# **As we can see there are lots of null values present in the dataset, we will check them in detail while performing Data Analysis and EDA.**

# ### "SalePrice" is the target variable, lets see how it looks like!!

# In[ ]:


# lets check the target variable "SalePrice"
house["SalePrice"].describe()


# **Great!! There are no negative values in the dataset for sale price which is good**

# In[ ]:


# lets check the distribution of saleprice
sns.distplot(house.SalePrice)


# **Saleprice seems to be skewed, This need to be handled else this will adversly impact our model**

# In[ ]:


# lets drop Id because its of no use to us
house.drop("Id",1,inplace = True)


# In[ ]:


# Let's display the variables with more than 0 null values
null_cols = []
for col in house.columns:
    if house[col].isnull().sum() > 0 :
        print("Column",col, "has", house[col].isnull().sum(),"null values")    
        null_cols.append(col)


# In[ ]:


# lets visualize the null vaues
plt.figure(figsize=(12,10))
sns.barplot(x=house[null_cols].isnull().sum().index, y=house[null_cols].isnull().sum().values)
xticks(rotation=45)
plt.show()


# **So many Null Values!!!**
# 
# **Let's take a look at the data dictionary, these are not actually the null values, rather these are the features which are not present in the house.**
# 
# **For example, let check the field Alley, Value "NA: here means house has "No Alley Access"**

# In[ ]:


# lets check if these null values actually have any relation with the target variable

house_eda = house.copy()

for col in null_cols:
    house_eda[col] = np.where(house_eda[col].isnull(), 1, 0)  

# lets see if these null values have to do anything with the sales price
plt.figure(figsize = (16,48))
for idx,col in enumerate(null_cols):
    plt.subplot(10,2,idx+1)
    sns.barplot(x = house_eda.groupby(col)["SalePrice"].median(),y =house_eda["SalePrice"])
plt.show()


# **From the above graphs, we can clearly see that the null values have strong relation with the SalePrice, hence we can niether drop the columns with null values, nor we can drop the rows with null values.**

# # Null Values Treatment

# In[ ]:


# all missing values for the categorical columns will be replaced by "None"
# all missing values for the numeric columns will be replaced by median of that field

for col in house.columns:
    if house[col].dtypes == 'O':
        house[col] = house[col].replace(np.nan,"None")
    else:
        house[col] = house[col].replace(np.nan,house[col].median())


# **There are some Date Variables in the dataset when we performed df.head(),Let's check again**

# In[ ]:


# making list of date variables
yr_vars = []
for col in house.columns:
    if "Yr" in col or "Year" in col:
        yr_vars.append(col)

yr_vars = set(yr_vars)
yr_vars


# **Let's check relation of these fields with the target variable**

# In[ ]:


plt.figure(figsize = (15,12))
for idx,col in enumerate(yr_vars):
    plt.subplot(2,2,idx+1)
    plt.plot(house.groupby(col)["SalePrice"].median())
    plt.xlabel(col)
    plt.ylabel("SalePrice")


# **Make a note of the trend of sale price with the field "YrSold", it shows a decreasing trend which seems unreal in real state scenario, price is expected to increase as the time passes by, but here it shows opposite. Let's handle this by creating "Age" variables from these variables**

# In[ ]:


# creating age variables
house['HouseAge'] =  house['YrSold'] - house['YearBuilt']
# age of master after remodelling
house['RemodAddAge'] = house['YrSold'] - house['YearRemodAdd']
# creating age of the garage from year built of the garage to the sale of the master
house['GarageAge'] = house['YrSold'] - house['GarageYrBlt'] 

# lets drop original variables
house.drop(["YearBuilt","YearRemodAdd","GarageYrBlt"],1,inplace = True)


# # Check variation in the feature values

# In[ ]:


# lets firs create seperate lists of categorical and numeric columns
cat_vars = []
num_vars = []
for col in house.columns.drop("SalePrice"):
    if house[col].dtypes == 'O':
        cat_vars.append(col)
    else:
        num_vars.append(col)

#lets check the lists created.
print("List of Numeric Columns:",num_vars)
print("\n")
print("List of Categorical Columns:",cat_vars)


# In[ ]:


# Let's further seperate the numeric features into continous and discrete numeric features
num_cont = []
num_disc = []
for col in num_vars:
    if house[col].nunique() > 25: # if variable has more than 25 different values, we consider it as continous variable
        num_cont.append(col)
    else:
        num_disc.append(col)


# In[ ]:


# lets check for the variance in the different continous numeric columns present in the dataset
house.hist(num_cont,bins=50, figsize=(20,15))
plt.tight_layout(pad=0.4)
plt.show()


# In[ ]:


# lets check the variance in numbers
for col in num_cont:
    print(house[col].value_counts())
    print("\n")


# **Following variables seems to have low variance:**
# * MasVnArea
# * BsmtFinSF2
# * 2ndFlrSF
# * EnclosedPorch
# * ScreenPorch

# In[ ]:


# lets check for the variance in the different discrete numeric columns present in the dataset
plt.figure(figsize = (16,96))
for idx,col in enumerate(num_disc):
    plt.subplot(9,2,idx+1)
    ax=sns.countplot(house[col])
    #for p in ax.patches:
    #    ax.annotate(p.get_height(), (p.get_x()+0.1, p.get_height()+10))


# **following variables seems to have low variance:**
# * **LowQualFinSF**
# * **BsmtHalfBath**
# * **KitchenAbvGr**
# * **3SsnPorch**
# * **PoolArea**
# * **MiscVal**
# 

# In[ ]:


# lets check for the variance in the categorical columns present in the dataset
plt.figure(figsize = (20,200))
for idx,col in enumerate(cat_vars):
    plt.subplot(22,2,idx+1)
    ax=sns.countplot(house[col])
    xticks(rotation=45)
    #for p in ax.patches:
    #    ax.annotate(p.get_height(), (p.get_x()+0.25, p.get_height()+5))


# In[ ]:


# lets check the variance in numbers
for col in cat_vars:
    print(house[col].value_counts())
    print("\n")


# **Following variables seems to have low variance:**
# 
# * MSZoning
# * Street,
# * Alley
# * LandContour,
# * Utilities,
# * LotConfig
# * Condition1
# * LandSlope
# * Condition2,
# * BldgType
# * RoofStyle
# * RoofMatl
# * ExterCond
# * BsmtCond
# * BsmtFinType2
# * Heating
# * CentralAir
# * Electrical
# * Functional
# * GarageQual
# * GarageCond
# * PavedDrive
# * PoolQC
# * Fence
# * MiscFeature
# * SaleType
# * SaleCondition

# In[ ]:


# lets drop the variables identified above as they have low variance
low_var_num_cont = ['MasVnrArea','BsmtFinSF2','2ndFlrSF','EnclosedPorch','ScreenPorch']

low_var_num_disc = ['LowQualFinSF','BsmtHalfBath','KitchenAbvGr','3SsnPorch','PoolArea','MiscVal']

low_var_cat_vars = ['MSZoning','Alley','LandContour','Utilities','LotConfig','Condition1','LandSlope','Condition2','BldgType','RoofStyle','RoofMatl','ExterCond','BsmtCond','BsmtFinType2','Heating','CentralAir','Electrical','Functional','GarageQual','GarageCond','PavedDrive','PoolQC','SaleType','SaleCondition','Street','Fence','MiscFeature']

house.drop(low_var_num_cont,1,inplace= True)
house.drop(low_var_num_disc,1,inplace= True)
house.drop(low_var_cat_vars,1,inplace= True)

num_cont = list(set(num_cont)-set(low_var_num_cont))
num_disc = list(set(num_disc)-set(low_var_num_disc))
cat_vars = list(set(cat_vars)-set(low_var_cat_vars))
       
num_vars = num_cont + num_disc


# # Lets handle Skewness before moving to Bi-Variate Analysis

# In[ ]:


# lets handle skewness in saleprice, lets take log to get normal distribution
house.SalePrice = np.log(house.SalePrice)
 
# lets check the distribution of saleprice again
sns.distplot(house.SalePrice)


# SalePrice looks good now, lets handle other numeric variables

# In[ ]:


# taking the log of numeric variables to hanlde skewness
num_features=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea']
for col in num_features:
    house[col] = np.log(house[col])


# # Bi-Variate analysis with "SalePrice"

# **Now we will see how SalePrice varies with respect to "Continous numeric variables" in the dataset**

# In[ ]:


# now lets plot the graphs for continous variables
plt.figure(figsize=(16,48))
for idx,col in enumerate(num_cont):
    plt.subplot(7,2,idx+1)
    plt.scatter(x = house[col],y=house["SalePrice"])
    plt.ylabel("SalePrice")
    plt.xlabel(col)


# **Most of the features above seems to have a good relation with SalePrice**
# 
# **There are some outliars present which need to be treated**

# **Now we will see how SalePrice varies with respect to "Discrete numeric variables" in the dataset**

# In[ ]:


# now lets plot the graphs for discrete variables
plt.figure(figsize=(16,48))
for idx,col in enumerate(num_disc):
    plt.subplot(10,2,idx+1)
    sns.boxplot(x = house[col],y=house["SalePrice"])
    plt.ylabel("SalePrice")
    plt.xlabel(col)


# We can drop MSSubClass, YrSold & MoSold as they have no impact on SalePrice

# In[ ]:


# dropping the variables
house.drop(['MSSubClass','YrSold','MoSold'],1,inplace= True)

num_disc = list(set(num_disc)-set(['MSSubClass','YrSold','MoSold']))
num_vars = list(set(num_vars)-set(['MSSubClass','YrSold','MoSold']))


# In[ ]:


# lets check relation of sale price with categorical variables
plt.figure(figsize=(16,48))
for idx,col in enumerate(cat_vars):
    plt.subplot(10,2,idx+1)
    sns.boxplot(x = house[col],y=house["SalePrice"])
    xticks(rotation=45)
    plt.ylabel("SalePrice")
    plt.xlabel(col)


# ### Outliars Detection

# In[ ]:


# lets create boxplots to detect outliars detection 
plt.figure(figsize=(16,48))
for idx,col in enumerate(num_vars):
    plt.subplot(11,2,idx+1)
    plt.boxplot(house[col])
    plt.xlabel(col)


# There are outliers in the dataset, these will be treated in the data engineering section

# In[ ]:


#train_set["SalePrice"] = train_sp
# lets check the variables
#num_vars = []
#for col in train_set.columns:
#    if train_set[col].dtypes != 'O':
#        num_vars.append(col)

for col in num_vars:
    print(house[col].describe(percentiles = [0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99]))
    print("\n")
    
# lets handle the outliers
q3 = house['OpenPorchSF'].quantile(0.99)
house = house[house.OpenPorchSF <= q3]
    
q3 = house['GarageArea'].quantile(0.99)
house = house[house.GarageArea <= q3]

q3 = house['TotalBsmtSF'].quantile(0.99)
house = house[house.TotalBsmtSF <= q3]

q3 = house['BsmtUnfSF'].quantile(0.99)
house = house[house.BsmtUnfSF <= q3]

q3 = house['WoodDeckSF'].quantile(0.99)
house = house[house.WoodDeckSF <= q3]

q3 = house['BsmtFinSF1'].quantile(0.99)
house = house[house.BsmtFinSF1 <= q3]


# In[ ]:


house.shape


# # Feature Engineering on Test Data Set

# In[ ]:


# lets read the test dataset, we will apply all the feature engineering operations on test set as well
test_set = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

# save "Id" in a variable and drop the column (as we have already dropped from train dataset)
test_set_id = test_set.Id
test_set.drop("Id",1,inplace = True)

# save SalePrice to a variable and drop it from training dataset as test dataset does not have this column
train_sp = house.SalePrice
house.drop("SalePrice",1,inplace=True)

# all missing values for the categorical columns will be replaced by "None"
# all missing values for the numeric columns will be replaced by median of that field
for col in test_set.columns:
    if test_set[col].dtypes == 'O':
        test_set[col] = test_set[col].replace(np.nan,"None")
    else:
        test_set[col] = test_set[col].replace(np.nan,test_set[col].median())


# creating age of the master from year built to the sale of the master
test_set['HouseAge'] =  test_set['YrSold'] - test_set['YearBuilt']
# age of master after remodelling
test_set['RemodAddAge'] = test_set['YrSold'] - test_set['YearRemodAdd']
# creating age of the garage from year built of the garage to the sale of the master
test_set['GarageAge'] = test_set['YrSold'] - test_set['GarageYrBlt'] 

# lets drop original variables
test_set.drop(["YearBuilt","YearRemodAdd","GarageYrBlt"],1,inplace = True)
        
        
# skewness in test set
# taking the log of numeric variables to hanlde skewness
num_features=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea']
for col in num_features:
    test_set[col] = np.log(test_set[col])

            
test_set.drop(low_var_num_cont,1,inplace= True)
test_set.drop(low_var_num_disc,1,inplace= True)
test_set.drop(low_var_cat_vars,1,inplace= True)

test_set.drop(['MSSubClass','YrSold','MoSold'],1,inplace= True)        
        

# merge the two datasets
master=pd.concat((house,test_set)).reset_index(drop=True)


# In[ ]:


master.shape


# In[ ]:


# In order to perform linear regression, we need to convert categorical variables to numeric variables.

# We have ordinal variables present in the dataest, lets treat them first:
master['ExterQual'] = master['ExterQual'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'None':0})
master['BsmtQual'] = master['BsmtQual'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'None':0})
master['BsmtExposure'] = master['BsmtExposure'].map({'Gd':4,'Av':3,'Mn':2,'No':1,'None':0})
master['BsmtFinType1'] = master['BsmtFinType1'].map({'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'None':0})
master['HeatingQC'] = master['HeatingQC'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'None':0})
master['KitchenQual'] = master['KitchenQual'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'None':0})
master['GarageFinish'] = master['GarageFinish'].map({'Fin':3,'RFn':2,'Unf':1,'None':0})
master['FireplaceQu'] = master['FireplaceQu'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'None':0})


# In[ ]:


# now lets create dummy variables for the remaining cateogorical variables
cat_vars = []
for col in master.columns:
    if master[col].dtypes == 'O':
        cat_vars.append(col)

# convert into dummies
master_dummies = pd.get_dummies(master[cat_vars], drop_first=True)

# drop categorical variables 
master.drop(cat_vars,1,inplace = True)

# concat dummy variables with X
master = pd.concat([master, master_dummies], axis=1)

# lets check the shape of the final dataset
master.shape


# In[ ]:


# we have perfomed all the necessary operations on the train and test datasets, time to sperate the two sets again
train_set = master[:1372]

test_set = master[1372:]


# # Data Scaling

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

y = train_sp.reset_index(drop=True)

scaler.fit(train_set)
X = scaler.transform(train_set)

# transform the train and test set, and add on the Id and SalePrice variables
X = pd.DataFrame(X,columns = train_set.columns).reset_index(drop=True)
X.head()

scaler.fit(test_set)
test_set = scaler.transform(test_set)
test_set = pd.DataFrame(test_set,columns = train_set.columns).reset_index(drop=True)


# # PCA  (Dimensionality reduction)

# In[ ]:


#Improting the PCA module
from sklearn.decomposition import PCA
pca = PCA(svd_solver='randomized', random_state=42)


# In[ ]:


X.isnull().sum().sort_values(ascending = False)


# In[ ]:


#let's apply PCA
pca.fit(X)


# In[ ]:


X.isnull().sum()


# In[ ]:


#List of PCA components.It would be the same as the number of variables
pca.components_


# In[ ]:


#Let's check the variance ratios
pca.explained_variance_ratio_


# In[ ]:


#Plotting the scree plot
#Making the screeplot - plotting the cumulative variance against the number of components
get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (12,8))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


# **From Scree plot we can conclude that we 60 PCs can explain around 90% variation of the dataset**

# In[ ]:


#Using incremental PCA for efficiency - saves a lot of time on larger datasets
from sklearn.decomposition import IncrementalPCA
pca_final = IncrementalPCA(n_components=60)


# In[ ]:


df_pca = pd.DataFrame(pca_final.fit_transform(X))
df_pca.shape


# In[ ]:


df_pca.head()


# # Linear Regression Model

# In[ ]:


import statsmodels.api as sm


# In[ ]:


# Add a constant to get an intercept
X_train_sm = sm.add_constant(df_pca)

# train the model
lr = sm.OLS(y, X_train_sm).fit()


# In[ ]:


# Performing a summary operation lists out all the different parameters of the regression line fitted
print(lr.summary())


# In[ ]:


# prediction on training dataset
y_train_pred = lr.predict(X_train_sm)


# In[ ]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[ ]:


r_squared = r2_score(y_train_pred, y)
r_squared


# **We have got 88% accuracy on the training dataset**

# **lets check RMSE as this is used by the kaggle competition for to evaluate model's predictive power**

# In[ ]:


from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y, y_train_pred))
rms


# In[ ]:


# lets make predictions on the test dataset

test_pca = pd.DataFrame(pca_final.fit_transform(test_set))

test_pca_sm = sm.add_constant(test_pca)
y_test_pred = lr.predict(test_pca_sm)


# Linear Regression produced good resuls, but, lets try Random Forest as well

# # Random Forest Model

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

# training the model
regr = RandomForestRegressor(n_estimators=50,random_state=0,n_jobs=1)
regr.fit(df_pca,y)


# In[ ]:


# lets make prediction on training dataset
y_train_pred = regr.predict(df_pca)


# In[ ]:


r_squared = r2_score(y_train_pred, y)
r_squared


# In[ ]:


from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y, y_train_pred))
rms


# **Clearly, performance is better with Random Forest, lets make final submission with RF model**

# In[ ]:


# lets make prediction on test dataset
y_test_pred = regr.predict(test_pca)


# In[ ]:


# lets prepare for the prediction submission
sub = pd.DataFrame()
sub['Id'] = test_set_id
sub['SalePrice'] = np.exp(y_test_pred)
sub.to_csv('submission.csv',index=False)


# In[ ]:





# In[ ]:




