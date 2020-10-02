#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


data = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')


# In[ ]:


data.shape


# In[ ]:


#categorical vars with na value
vars_with_na = [var for var in data.columns if data[var].isnull().sum()>1 and data[var].dtypes=='O']
Null_List =[]
for var in vars_with_na:
    Null_List.append([var,np.round(data[var].isnull().mean(),3)])


# In[ ]:


Null_List


# In[ ]:


def  RemoveHighNA(Null_List):
    
    NullvalueResult =  []
    for var in Null_List:
        if var[1] > 0.25:
            NullvalueResult.append(var[0])
            
            #print(var)
    return NullvalueResult     


# In[ ]:


result = RemoveHighNA(Null_List)
print(result)


# In[ ]:


data = data.drop(result,axis = 1)


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

# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)


# In[ ]:


X_train , X_test ,y_train ,y_test = train_test_split(data,data.SalePrice,test_size = 0.1,random_state = 0)
X_train.shape,X_test.shape


# In[ ]:





# In[ ]:


y_actual_val = y_test


# In[ ]:


y_train.head()


# In[ ]:


X_train.head()


# In[ ]:


vars_with_na = [var for var in X_train.columns if X_train[var].isnull().sum()>1 and X_train[var].dtypes=='O']


# In[ ]:


vars_with_na


# In[ ]:


#Fill categorical variables with na values 
def fill_categorical_na(df,var_list):
    X = df.copy()
    X[var_list] = df[var_list].fillna('Missing')
    return X


# In[ ]:


X_train = fill_categorical_na(X_train,vars_with_na)
X_test = fill_categorical_na(X_test , vars_with_na)

#check if missing values are still tehre 
X_train[vars_with_na].isnull().sum()


# In[ ]:


# check that test set does not contain null values in the engineered variables
[vr for var in vars_with_na if X_train[var].isnull().sum()>0]


# In[ ]:


#Make a list of numerical variables that have missing values 
vars_with_na = [var for var in data.columns if X_train[var].isnull().sum()>1 and X_train[var].dtypes!='O']

#print varibles and missing values 
for var in vars_with_na:
    print(var, np.round(X_train[var].isnull().mean(), 3))


# In[ ]:


#replace missing values 
for var in vars_with_na:
    #caliculate mode 
    mode_val = X_train[var].mode()[0]
    
    #train 
    #X_train[var+'_na'] = np.where(X_train[var].isnull(),0,1)
    X_train[var].fillna(mode_val,inplace = True)
    # test
    #X_test[var+'_na'] = np.where(X_test[var].isnull(), 0, 1)
    X_test[var].fillna(mode_val, inplace=True)

# check that we have no more missing values in the engineered variables
X_train[vars_with_na].isnull().sum()


# In[ ]:


# check that we have the added binary variables that capture missing information
#X_train[['LotFrontage_na', 'MasVnrArea_na', 'GarageYrBlt_na']].head()


# In[ ]:


# check that test set does not contain null values in the engineered variables
[vr for var in vars_with_na if X_test[var].isnull().sum()>0]


# > X_train , X_test ,y_train ,y_test****

# In[ ]:


#Lets engineer relation between year var and house price 
def elapsed_years(df, var):
    # capture difference between year variable and year the house was sold
    df[var] = df['YrSold'] - df[var]
    return df


# In[ ]:


ListvarsYears = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']


# In[ ]:


for var in ListvarsYears:
    X_train = elapsed_years(X_train, var)
    X_test = elapsed_years(X_test, var)


# In[ ]:


#verify the same data 
# check that test set does not contain null values in the engineered variables
[vr for var in ListvarsYears if X_test[var].isnull().sum()>0]


# In[ ]:


[vr for var in ListvarsYears if X_train[var].isnull().sum()>0]


# In[ ]:


vars_with_numbers_log = [var for var in data.columns if  X_train[var].dtypes!='O' and var != "SalePrice" and var != 'Id' and var != 'YearBuilt' and var != 'YearRemodAdd' and var != 'GarageYrBlt' and var != 'YrSold' ]


# In[ ]:


len(vars_with_numbers_log)


# In[ ]:


vars_with_numbers_log


# In[ ]:





# In[ ]:


vars_with_numbers_log


# log transform all the numerical varibles 

# In[ ]:


#listnumericalvars =['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MiscVal', 'SalePrice']
#listnumericalvars = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']


# In[ ]:


def diaognizeplots(df,variable):
  
    plt.figure(figsize=(15,6))
    plt.subplot(1,2,1)
    df[variable].hist(bins = 30)
    plt.subplot(1,2,2)
    stats.probplot(df[variable],dist = "norm",plot=plt)
    plt.show()
for var in vars_with_numbers_log:
    if var != "Id":
        diaognizeplots(X_train,var)


# I am using equal frequency discretisation from sikitlearn  

# In[ ]:


from sklearn.preprocessing import KBinsDiscretizer


# In[ ]:


X_train["LotFrontage"]


# In the following code i am using kbins tech to see if  iam able to remove outliers 

# In[ ]:


from sklearn.preprocessing import KBinsDiscretizer
LotFrontage_Tran  = KBinsDiscretizer(n_bins=5,encode='ordinal',strategy='kmeans')
LotFrontage_Tran.fit(X_train[vars_with_numbers_log])


# In[ ]:


Exppdescritizer = LotFrontage_Tran.transform(X_train[vars_with_numbers_log])
Exppdescritizer = pd.DataFrame(Exppdescritizer,columns=vars_with_numbers_log)


# In[ ]:


Exppdescritizer.head()


# In[ ]:



for var in Exppdescritizer:
    if var != "Id":
        print("Current plt =",var)
        diaognizeplots(X_train,var)
        diaognizeplots(disckbins,var)


# K bins did not work out .. 
# I am trying n bins approach

# In[ ]:


from sklearn.preprocessing import KBinsDiscretizer
disckbins  = KBinsDiscretizer(n_bins=10,encode='ordinal',strategy='quantile')
disckbins.fit(X_train[vars_with_numbers_log])


# In[ ]:


print(type(vars_with_numbers_log))


# In[ ]:


disckbins = disckbins.transform(X_train[vars_with_numbers_log])
disckbins = pd.DataFrame(disckbins,columns=vars_with_numbers_log)


# In[ ]:


for var in disckbins:
    if var != "Id":
        print("Current plt =",var)
        diaognizeplots(X_train,var)
        diaognizeplots(disckbins,var)


# Log did not work fine .. I will use a different function .. aim is to clear outlier first

# In[ ]:


#Engineer the numerical varibles 
for var in vars_with_numbers_log:
    if 0 in data[var].unique():
        pass
    else:
        #df[var] = np.log(df[var])
        X_train[var] = np.log(X_train[var])
        X_test[var]= np.log(X_test[var])


# check this step again 

# In[ ]:


def diaognizeplots(df,variable):
    print("Current plt =",variable)
    plt.figure(figsize=(15,6))
    plt.subplot(1,2,1)
    df[variable].hist(bins = 30)
    plt.subplot(1,2,2)
    stats.probplot(df[variable],dist = "norm",plot=plt)
    plt.show()
for var in vars_with_numbers_log:
    if var != "Id":
        diaognizeplots(X_train,var)


# In[ ]:


#checking null values 
[var for var in vars_with_numbers_log if X_test[var].isnull().sum()>0]


# In[ ]:


#same for train 
[var for var in vars_with_numbers_log if X_train[var].isnull().sum()>0]


# In[ ]:


X_train.columns


# In[ ]:


#catogerical varibles 
#lets remove values that have less than 1 % values 
cat_vars = [var for var in X_train.columns if X_train[var].dtype == 'O']
cat_vars


# In[ ]:


def find_frequent_labels(df,var,rare_prec):
    #find labels tht are shared by more than one percent of data 
    df = df.copy()
    tmp = df.groupby(var)['SalePrice'].count()/len(df)
    return tmp[tmp>rare_prec].index
for var in cat_vars:
    freuent_ls = find_frequent_labels(X_train,var,0.01)
    X_train[var] = np.where(X_train[var].isin(freuent_ls),X_train[var],'Rare')
    X_test[var] = np.where(X_test[var].isin(freuent_ls),X_test[var],'Rare')
    


# Look back again 

# In[ ]:


#this functin will assign descrete varible
#Smaller value corrensponds to smaller mean of the target 
def replace_categorical_vars(train,test,var,target):
    extraindex = train.groupby([var])[target].mean().sort_values()
    print("Extra index ++++++++++++++++++",extraindex)
    ordered_labels = train.groupby([var])[target].mean().sort_values().index
    print(ordered_labels)
    ordinal_label  = {k:i for i,k in enumerate(ordered_labels,0)}
    print("Second print==================",ordinal_label)
    train[var]=train[var].map(ordinal_label)
    test[var] = test[var].map(ordinal_label)


# In[ ]:


for var in cat_vars:
    replace_categorical_vars(X_train,X_test,var,'SalePrice')
    


# In[ ]:


#check for absence of na 
[var for var in X_train.columns if X_train[var].isnull().sum()>0]


# In[ ]:


[var for var in X_test.columns if X_test[var].isnull().sum()>0]


# In[ ]:


#analyse catogorical varibles 
def analyse_vars(df,var):
    df = df.copy()
    df.groupby(var)['SalePrice'].median().plot.bar(color = ['r','b','g','y'])
    plt.title(var)
    plt.ylabel('SalePrice')
    plt.show()
for var in cat_vars:
    analyse_vars(X_train,var)


# In[ ]:


#Lets do some feature scaling 
len(X_train.columns)


# In[ ]:


train_vars = [var for var in X_train.columns if var not in ['Id']]
len(train_vars)


# In[ ]:


X_train['Id']


# In[ ]:



#X_train.drop('SalePrice',axis=1,inplace=True)
#X_test.drop('SalePrice',axis=1,inplace=True)


# In[ ]:


#X_train.drop('Id',axis=1,inplace=True)
#X_test.drop('Id',axis=1,inplace=True)


# In[ ]:


X_train.head()


# In[ ]:


#for val in X_train.columns:
#    print(val)
 #   print(X_train[var].dtypes())
#np.max(X_test.SalePrice)
    


# In[ ]:


#for val in train_vars:
    #print(X_train[val].dtypes)
#print(len(train_vars))
for val in train_vars:
    print(val)
    for temp in val:
        if temp == '-inf' or temp == 'nan':
            print(temp)


# In[ ]:


X_train[train_vars].iloc[10]


# In[ ]:


X_train['MiscVal']


# In[ ]:


print(X_train.head())


# In[ ]:


train_vars


# In[ ]:


def diaognizeplots(df,variable):
    print("Current plt =",variable)
    plt.figure(figsize=(15,6))
    plt.subplot(1,2,1)
    df[variable].hist(bins = 30)
    plt.subplot(1,2,2)
    stats.probplot(df[variable],dist = "norm",plot=plt)
    plt.show()
for var in X_train.columns:
    if var != "Id":
        diaognizeplots(X_train,var)


# In[ ]:


print(max(X_test['SalePrice']))
print(min(X_test['SalePrice']))
Max_sale = max(X_test['SalePrice'])
Min_sale = min(X_test['SalePrice'])


# In[ ]:


#fit Scalar 
scaler  = MinMaxScaler() #ceate instance 
scaler.fit(X_train[train_vars]) #  fit  the scaler to the train set for later use

scalertwo  = MinMaxScaler() #ceate instance 
scalertwo.fit(X_test[train_vars]) 

# transform the train and test set, and add on the Id and SalePrice variables

X_train = pd.concat([X_train[['Id']].reset_index(drop=True),
                    pd.DataFrame(scaler.transform(X_train[train_vars]), columns=train_vars)],
                    axis=1)
#train_vars.append




X_test = pd.concat([X_test[['Id']].reset_index(drop=True),
                  pd.DataFrame(scalertwo.transform(X_test[train_vars]), columns=train_vars)],
                 axis=1)
y_train = X_train['SalePrice']
y_test = X_test['SalePrice']
#resactual = scalertwo.inverse_transform(X_test[train_vars])

X_train.drop('SalePrice',axis=1,inplace=True)
X_test.drop('SalePrice',axis=1,inplace=True)
train_vars.remove('SalePrice')


# In[ ]:





# In[ ]:


X_train.head()


# In[ ]:


#checking for  outliar varibles 
def find_outliers(df,var):
    df = df.copy()
    if 0 in data[var].unique():
        pass
    else:
        #df[var] = np.log(df[var])
        df.boxplot(column= var)
        plt.title(var)
        plt.ylabel(var)
        plt.show()
for var in X_train.columns:
    if var != 'LotFrontage_na':
        find_outliers(X_train,var)


# Lets do some detialed plots the above ones are not that great 
# 

# In[ ]:


import scipy.stats as stats


# In[ ]:





# In[ ]:





# In[ ]:


#X_train.SalePrice.hist(bins=100)vgjk


# In[ ]:


m,n.


# In[ ]:


print(y_train.head())


# In[ ]:


X_train.head()    
    


# fix the outliers 

# In[ ]:


X_train.isnull().sum()


# In[ ]:


#Saving the new data that has been transformed 
X_train.to_csv('xtrain_tran.csv', index=False)
X_test.to_csv('xtest_tran.csv', index=False)


# In[ ]:





# In[ ]:


rangeone = Max_sale- Min_sale 


# In[ ]:


X_train.head()


# In[ ]:


y_train.head()


# In[ ]:


from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(X_train, y_train,early_stopping_rounds=5, 
             eval_set=[(X_train, y_train)], 
             verbose=True)


# In[ ]:


from sklearn.metrics import mean_absolute_error

predictions = my_model.predict(X_test)
print(predictions)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_test)))


# In[ ]:


print(type(predictions))


# In[ ]:


predact = []
for temp in predictions:
    resulttemp = temp*rangeone+Min_sale
    predact.append(resulttemp)


# In[ ]:


actval = []
for temp in y_test:
    resulttemp = temp*rangeone+Min_sale
    actval.append(resulttemp)


# In[ ]:


print(len(y_test))


# In[ ]:


leny = len(y_test)
resavg = 0 
for i,j,k,l in zip(predact,actval,predictions,y_test):
    print(int(i),'values',j,"difference = ",((int(i)-j)*100)/i)
    tempavgval = (((int(i)-j)*100)/i)
    resavg = resavg+tempavgval
    
    #print(k,'second print',l,"difference = ",((k-l)*100)/k)
print("the difference val ============",resavg/leny)


# In[ ]:




