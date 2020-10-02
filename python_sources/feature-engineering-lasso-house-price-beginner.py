#!/usr/bin/env python
# coding: utf-8

# In this kernel we will try to do a feature engineering on this data set.This data set is a work in process.If you like my work please do vote.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ### Importing Python Modules 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.pandas.set_option('display.max_columns',None)


# In[ ]:


data=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
print(data.shape)
data.head()


# ## A]Data Analysis
# ### We will analyse the dataset to identify
# 
# 1.Missing values
# 
# 2.Numerical variables 
# 
# 3.Distribution of numerical variables 
# 
# 4.Outliers 
# 
# 5.Categorical Variables
# 
# 6.Cardinality of categorical variables 
# 
# 7.Potential relationship between variables and target (SalePrice)

# ### 1.Missing values

# In[ ]:


#make a list of the variables that contain missing values
vars_with_na=[var for var in data.columns if data[var].isnull().sum()>1]

#print the variable name and the percentage of missing values 
for var in vars_with_na:
    print(var,np.round(data[var].isnull().mean(),3),'% missing values')


# Many of our columns have missing values.We will be dealing with the missing values in the coming steps.

# ### 2.Relationship between values being missing and Sale Price 

# In[ ]:


def analyse_na_value(df,var):
    df=df.copy()
    
    #Let's make a variable that indicates 1 if the observation was missing or Zero otherwise 
    df[var]=np.where(df[var].isnull(),1,0)
    
    #Let's calculate the mean SalePrice where the information is missing or present 
    df.groupby(var)['SalePrice'].median().plot.bar()
    plt.title(var)
    plt.show()
    
for var in vars_with_na:
    analyse_na_value(data,var)


# 0 -Means value is available and 1- Means Value is missing.

# ### 3.Numerical Variables
# 
# Lets go ahead and find out what numerical values we have in the dataset

# In[ ]:


# List of numerical variables
num_vars= [var for var in data.columns if data[var].dtypes!='O' ]

print('Number of numerical variables: ',len(num_vars))

# Visualise the numerical variables 
data[num_vars].head()


# In[ ]:


print('Number of House Id labels:',len(data.Id.unique()))
print('Number of Houses in the Dataset',len(data))


# So there are 38 columns with numerical Values.We wont be needing the colum Id for makinf predictions for SalePrice of house.

# ### 4.Temporal variables
# 
# We can notice that there are four columns with the year data.We generally dont use this information directly to predict the SalePrice.We will do some feature engineering like difference between the year of built and the year when the house was sold.

# In[ ]:


# List of Variables that contain year information 
year_vars=[var for var in num_vars if 'Yr' in var or 'Year' in var]
year_vars


# In[ ]:


# Lets explore the content of the years variables
for var in year_vars:
    print(var,data[var].unique())


# In[ ]:


# Evloution of House Price with Year
data.groupby('YrSold')['SalePrice'].median().plot()
plt.ylabel('Median House Price')
plt.title('Change in House Price with years');


# We can see an unsual thing that the House Price is decreasing in year 2008-09.This was due to financial crisis that happened.

# In[ ]:


# Lets's explore the relationship between the year variable and the house price in little more details
def analyse_year_vars(df,var):
    df=df.copy()
    
    # capture differnce between year variable and year the house was sold 
    df[var]=df['YrSold']-df[var]
    
    plt.scatter(df[var],df['SalePrice'])
    plt.ylabel('SalePrice')
    plt.xlabel(var)
    plt.show()

for var in year_vars:
    if var !='YrSold':
       analyse_year_vars(data,var)


# From all the above plots between the year and SalePrice data we can see that the price of the house decreases as it becomes older.So one of this variable is could be very useful in predicting the housing price.

# ### 5.Discrete Variables
# 
# Let's go ahead and find which variables are discrete ie show a finite number of values

# In[ ]:


# List of Discrete Variables 
discrete_vars=[var for var in num_vars if len(data[var].unique())<20 and var not in year_vars+['Id']]

print('Number of discrete variables:',len(discrete_vars))


# In[ ]:


# Let's visualize the discrete variables

data[discrete_vars].head()


# We can see that this variable tend to be Qualifications or grading scales or refer to number of rooms or units.Now lets go ahead and analyse their contribution to the SalePrice.

# In[ ]:


def analyse_discrete(df,var):
    df=df.copy()
    df.groupby(var)['SalePrice'].median().plot.bar()
    plt.title(var)
    plt.ylabel('SalePrice')
    plt.show()
    
for var in discrete_vars:
    analyse_discrete(data,var)


# We can go through the individual plots and derive our own conclusions.The relationship between the variable and the SalePrice is not always monotonic.For OverallQual there is a monotonic relationship the higher the quality the higher the SalePrice.

# ### 6.Continous Variables

# In[ ]:


# List of Continous variables 
cont_vars=[var for var in num_vars if var not in discrete_vars + year_vars+['Id']]

print('Number of continous variables:',len(cont_vars))


# In[ ]:


# Let's Visualize the continous variables

data[cont_vars].head()


# In[ ]:


# Lets Go ahead and analyse the distribution of this variables
def analyse_continous(df,var):
    df=df.copy()
    df[var].hist(bins=20)
    plt.ylabel('Number of houses')
    plt.xlabel(var)
    plt.title(var)
    plt.show()
    
for var in cont_vars:
    analyse_continous(data,var)


# We can see from the distribution of the variables that most of the distributions are scewed.For linear regression we assume that our vaiables have linear distribution.To get better prediction from our models we need to convert the variables into Normal Or Gaussian Distribution.To convert the variables into normal distribution we need to apply log function to the variables.

# In[ ]:


# Lets go ahead and analyse the distribution of this variables with log function
def analyse_transformed_continous(df,var):
    df=df.copy()
    
    # Log does not take negative value,so let's be careful and skip those variables 
    if 0 in data[var].unique():
        pass
    else:
        # Log transform the variable 
        df[var]=np.log(df[var])
        df[var].hist(bins=20)
        plt.ylabel('Number of houses')
        plt.xlabel(var)
        plt.title(var)
        plt.show()

for var in cont_vars:
    analyse_transformed_continous(data,var)


# So most of over variables have been transformed from skewed to normal Gaussian distrubution.This will help us to improve the prediction of results from our model.

# In[ ]:


# Lets explore the relationship between the transformed varibales and the house Sale Price 

def transform_analyse_continous(df,var):
    df=df.copy()
    
    # Log does not take negative values, so let's be careful and skip those variables 
    if 0 in data[var].unique():
        pass
    else:
        # Log transform
        df[var]=np.log(df[var])
        df['SalePrice']=np.log(df['SalePrice'])
        plt.scatter(df[var],df['SalePrice'])
        plt.ylabel('SalePrice')
        plt.xlabel(var)
        plt.show()
        
for var in cont_vars:
    if var !='SalePrice':
        transform_analyse_continous(data,var)


# So we can see that GrLivArea and 1stFirSF have good linear correlation with the Sale Price of the house.

# ### 7.Outliers

# In[ ]:


# Let's make boxplots to visualise outliers in the continous variables 

def find_outliers(df,var):
    df=df.copy()
    
    # Log does not take negative values,so let's be careful and skip those variables 
    if 0 in data[var].unique():
        pass
    else:
        df[var]=np.log(df[var])
        df.boxplot(column=var)
        plt.title(var)
        plt.ylabel(var)
        plt.show()
        
for var in cont_vars:
    find_outliers(data,var)


# All the variables have outliers.The perfromance of linear models is affected by outliers.Removing the outliers can improve the accuracy of the model.

# ### 8.Categorical variables

# In[ ]:


### Categorical variables 

cat_vars=[var for var in data.columns if data[var].dtype=='O']

print('Number of categorical variables:',len(cat_vars))


# In[ ]:


# Let's visualize the values of categorical variables 
data[cat_vars].head()


# ### 8.1 Number of labels:cardinality
# 
# Let's evaluate the number of categories present in each variable 

# In[ ]:


for var in cat_vars:
    print(var,len(data[var].unique()),'categories')


# All the variables have less number of categories within them.So we can say they have low cardinality.If the dataset has higher cardinality then we need to do feature engineering to improve our model accuracy.

# ### 8.2 Rare labels
# 
# Lets find out if there labels which are present for only few houses 

# In[ ]:


def analyse_rare_labels(df,var,rare_perc):
    df=df.copy()
    tmp=df.groupby(var)['SalePrice'].count()/len(df)
    return tmp[tmp<rare_perc]

for var in cat_vars:
    print(analyse_rare_labels(data,var,0.01))


# There are some labels in the dataset which have less than 1% contribution.They can cause problem with the Machine learning model as they may result in the overfitting of the model.It is better to remove such labels from the datset.

# ## B]Feature Engineering

# ### 1.Setting Seed

# In[ ]:


# to handle the dataset
import pandas as pd
import numpy as np

# for plotting 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# to divide train and test set
from sklearn.model_selection import train_test_split

# feature scaling 
from sklearn.preprocessing import MinMaxScaler

# to visualise all the columns in the dataframe
pd.pandas.set_option('display.max_columns',None)


# In[ ]:


data=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
print(data.shape)
data.head()


# ### 2.Separate Dataset into Train and Test 
# 
# Before beginning to engineer our features,it is important to separate our data into training and testing set.This is to avoid over-fitting.This step involves randomness so we need to set the seed.

# In[ ]:


# Let's Separate into train and test set
# Remember to se the seed (random_state for the sklearn function)

X_train,X_test,y_train,y_test=train_test_split(data,data.SalePrice,test_size=0.1,random_state=0) # Here we are setting the seed 


# ### 3.Missing values 
# 
# For categorical variables,we will fill the the missing information with a additional category 'missing'

# In[ ]:


#make a list of the variables that contain missing values
vars_with_na=[var for var in data.columns if X_train[var].isnull().sum()>1 and X_train[var].dtypes=='O']

#print the variable name and the percentage of missing values 
for var in vars_with_na:
    print(var,np.round(data[var].isnull().mean(),3),'% missing values')


# In[ ]:


# function to replace NA in categorical variabsables 
def fill_categorical_na(df,var_list):
    X=df.copy()
    X[var_list]=df[var_list].fillna('Missing')
    return X


# In[ ]:


# replace missing values with new label: "Missing"
X_train=fill_categorical_na(X_train,vars_with_na)
X_test=fill_categorical_na(X_test,vars_with_na)

# check that we have no missing information in the engineered variables 
X_train[vars_with_na].isnull().sum()


# In[ ]:


# check that the test set does not contain null values in the engineered variables
[vr for var in vars_with_na if X_test[var].isnull().sum()>0]


# For numerical variables,we are going to add an additional variable capturing the missing information and then replace the missing information in the original variable by the mode or most frequent value.  

# In[ ]:


# make a list of the numerical variables that contain missing values 
vars_with_na=[var for var in data.columns if X_train[var].isnull().sum()>1 and X_train[var].dtypes!='O']

# print the variable name and the percentage of missing values 
for var in vars_with_na:
    print(var,np.round(X_train[var].isnull().mean(),3),'% missing values')


# In[ ]:


# replace the missing value 
for var in vars_with_na:
    
    # calculate the mode
    mode_val= X_train[var].mode()[0]
    
    # train
    X_train[var+'_na']= np.where(X_train[var].isnull(),1,0)
    X_train[var].fillna(mode_val,inplace=True)
    
    # test
    X_test[var+'_na']= np.where(X_test[var].isnull(),1,0)
    X_test[var].fillna(mode_val,inplace=True)
    
# check that we have no more missing values in the engineering variables 
X_train[vars_with_na].isnull().sum()


# In[ ]:


X_train[['LotFrontage_na','MasVnrArea_na','GarageYrBlt_na']].head()


# In[ ]:


# check that the test set doesnt have null values in the engineered variables 
[vr for var in vars_with_na if X_test[var].isnull().sum()>0]


# ### 4.Temporal variables 
# 
# We have 4 variables that refer to the years in which something was built or something specific happened.We will capture the time elapsed between the variable and year thats house was sold.

# In[ ]:


# Let's explore the relationship between the year variables and the house price in bit more details

def elapsed_years(df,var):
    # capture difference between year variable and the year the house was sold
    df[var] = df['YrSold'] -df[var]
    return df


# In[ ]:


for var in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
    X_train = elapsed_years(X_train,var)
    X_test = elapsed_years(X_test,var)


# In[ ]:


# check that test set does not contain null values in the engineered variables 
[vr for var in ['YearBuilt','YearRemodAdd','GarageYrBlt'] if X_test[var].isnull().sum()>0]


# ### 5.Numerical variables
# 
# We will lofg transform the numerical variables that do not contain zeros in order to get a more Gausian-Like distribution.This tends to help Linear machine learning models.

# In[ ]:


for var in ['LotFrontage','LotArea','1stFlrSF','GrLivArea','SalePrice']:
    X_train[var]=np.log(X_train[var])
    X_test[var]=np.log(X_test[var])   


# In[ ]:


# check that the test set does not contain null values in the engineered variables 
[var for var in ['LotFrontage','LotArea','1stFlrSF','GrLivArea','SalePrice'] if X_test[var].isnull().sum()>0]


# In[ ]:


# check that the train set does not contain null values in the engineered variables 
[var for var in ['LotFrontage','LotArea','1stFlrSF','GrLivArea','SalePrice'] if X_train[var].isnull().sum()>0]


#  ### 6.Categorical variables
#  
#  First we will remove those categories within variables that are present in less than 1% of the observations

# In[ ]:


# Let's capture the categorical variables first 
cat_vars = [var for var in X_train.columns if X_train[var].dtype=='O']


# In[ ]:


def find_frequent_labels(df,var,rare_perc):
    # finds the labels that are shared by more than a certain % of the houses in the dataset 
    df=df.copy()
    tmp=df.groupby(var)['SalePrice'].count()/len(df)
    return tmp[tmp>rare_perc].index

for var in cat_vars:
    frequent_ls = find_frequent_labels(X_train,var,0.01)
    X_train[var] = np.where(X_train[var].isin(frequent_ls),X_train[var],'Rare')
    X_test[var] = np.where(X_test[var].isin(frequent_ls),X_test[var],'Rare')


# Next we need to transform the strings of these varibales into numbers.We will do it so that we capture the monotinic relationship between the label and the target.

# In[ ]:


# this function will assign discrete values to the strings of the variables,
# so that the similar value corresponds to the smaller mean target 

def replace_categories(train,test,var,target):
    ordered_labels=train.groupby([var])[target].mean().sort_values().index
    ordinal_label ={k:i for i,k in enumerate(ordered_labels,0)}
    train[var]=train[var].map(ordinal_label)


# In[ ]:


for var in cat_vars:
    replace_categories(X_train,X_test,var,'SalePrice')


# In[ ]:


# check absence of na
[var for var in X_train.columns if X_train[var].isnull().sum()>0]


# In[ ]:


# check absence of na
[var for var in X_test.columns if X_test[var].isnull().sum()>0]


# In[ ]:


# Let me show you what I mean by monotonic relationship between the labels and target
def analyse_vars(df,var):
    df=df.copy()
    df.groupby(var)['SalePrice'].median().plot.bar()
    plt.title(var)
    plt.ylabel('SalePrice')
    plt.show()
    
for var in cat_vars:
    analyse_vars(X_train,var)


# We can now see monotonic relationships between the labels of our variables and the target (remember that the target is log-transformed,that is why the difference seem so small)

# ### 7.Feature Scaling 
# 
# For use in linear models,features need to be either scaled or normalised.In the next section,I will scale features between the min and max values:

# In[ ]:


train_vars = [var for var in X_train.columns if var not in ['Id','SalePrice']]
len(train_vars)


# In[ ]:


X_train[['Id','SalePrice']].reset_index(drop=True)


# In[ ]:


# Fit scaler 
scaler = MinMaxScaler() # create an instance 
scaler.fit(X_train[train_vars])  # fit the scaler to the train set for later user 

# transform the train and test set, and add on the Id and the SalePrice variables 
X_train = pd.concat([X_train[['Id','SalePrice']].reset_index(drop=True),pd.DataFrame(scaler.transform(X_train[train_vars]),columns=train_vars)],axis=1)

X_test = pd.concat([X_test[['Id','SalePrice']].reset_index(drop=True),pd.DataFrame(scaler.transform(X_test[train_vars]),columns=train_vars)],axis=1)


# In[ ]:


X_train.head()


# In[ ]:


# Lets now save the train and test sets for the future reference
#X_train.to_csv('xtrain.csv',index=False)
#X_test.to_csv('xtest.csv',index=False)


# ### C]Feature Selection 

# In[ ]:


# to build the models 
from sklearn.linear_model import Lasso 
from sklearn.feature_selection import SelectFromModel


# In[ ]:


# capture the target 
y_train = X_train['SalePrice']
y_test = X_test['SalePrice']

# drop unnecessary variables from our training and testing sets 
X_train.drop(['Id','SalePrice'],axis=1,inplace=True)
X_test.drop(['Id','SalePrice'],axis=1,inplace=True)


# Let's go ahead and select a subset of the most predictive features.There is an element of randomness in the Lasso regression so rememeber to se the seed.

# In[ ]:


#
#
#

sel_ = SelectFromModel(Lasso(alpha=0.005,random_state=0))
sel_.fit(X_train,y_train)


# In[ ]:


# this command lets us visualise those feature that were kept 
# kept features are marked as True
sel_.get_support()


# In[ ]:


selected_feat=X_train.columns[(sel_.get_support())]

print('total features: {}'.format((X_train.shape[1])))
print('Selected features: {}'.format(len(selected_feat)))
print('Features with coefficients shrank to zero: {}'.format(np.sum(sel_.estimator_.coef_==0)))


# In[ ]:


selected_feat


# ### Identify the selected variables

# In[ ]:


selected_feat = X_train.columns[(sel_.estimator_.coef_!=0).ravel().tolist()]
selected_feat


# In[ ]:


#pd.Series(selected_feats).to_csv('selected_features.csv',index=False)


# ### D]Model building 

# #### Regularised Linear Regression 

# In[ ]:


from sklearn.metrics import mean_squared_error
from math import sqrt


# In[ ]:


lin_model = Lasso(alpha=0.005,random_state=0)
lin_model.fit(X_train,y_train)


# In[ ]:


pred=lin_model.predict(X_train)
print('linear train mse: {}'.format(mean_squared_error(np.exp(y_train),np.exp(pred))))
print('linear train rmse:{}'.format(sqrt(mean_squared_error(np.exp(y_train),np.exp(pred)))))
print()
#pred=lin_model.predict(X_test)
#print('linear train mse: {}'.format(mean_squared_error(np.exp(y_test),np.exp(pred))))
#print('linear train rmse:{}'.format(sqrt(mean_squared_error(np.exp(y_test),np.exp(pred)))))
#print()
print('Average house price:',np.exp(y_train).median())


# In[ ]:


# Let's evaluate our predictions wrt to original price 
#plt.scatter(y_test,lin_model.predict(X_test))
#plt.xlabel('True House Price')
#plt.ylabel('Predicted House Price')
#plt.title('Evaluation of Lasso Predictions')


# In[ ]:


# Let's evaluae the distrubution of the errors :
# They should be fairly normally distributed

#errors = y_test - lin_model.predict(X_test)
#errors.hist(bins=15)


# In[ ]:


# Feature importance 

"""importance = pd.Series(np.abs(lin_model.coef_.ravel()))
importance.index = selected_feat
importance.sort_values(inplace=True,ascending=False)
importance.plot.bar(figsize=(18,6))
plt.ylabel('Lasso Coefficents')
plt.title('Feature importance')"""


# In[ ]:




