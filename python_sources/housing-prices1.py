#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import RidgeCV, LinearRegression, LassoCV
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from IPython.display import display, HTML
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
############################## drop ID column #######################
ID = test['Id']
test.drop('Id', axis = 1, inplace = True)
train.drop('Id', axis = 1, inplace = True)

#####################################all_data##########################
all_data = pd.concat((train, test)).reset_index(drop = True)

# Any results you write to the current directory are saved as output.


# In[ ]:


sum(train['Condition2'] == train['Condition2'].mode()[0])


# In[ ]:


print(train.shape)
train.head()
297 in train.index


# In[ ]:


print(test.shape)
test.head()


# > ## Numerical Features

# In[ ]:


numeric_features = list(test.select_dtypes(exclude=["object",'category']).columns)
numeric_features_num = len(numeric_features)
print("no of numerical feature: ",numeric_features_num,"\nNumerical Features: ",
      numeric_features)


# In[ ]:


train.describe()


# In[ ]:


pd.options.display.max_columns = 45
test.describe()


# ## Categorical Features

# In[ ]:


category_feature = list(test.select_dtypes(exclude=[np.number]).columns)
category_feature_num = len(category_feature)
print("no. of categorical feature: ",category_feature_num,"\nCategorical Features: ", category_feature)


# In[ ]:


train.describe(include = ['O'])              #Shows Categorical Features


# In[ ]:


test.describe(include = ['O'])               #Shows Categorical Features


# ##  Saleprice

# In[ ]:


train["SalePrice"].describe()
SP = train["SalePrice"]
print(train['SalePrice'].iloc[297])


# In[ ]:


import scipy.stats as st
sns.distplot(train["SalePrice"], kde=True, fit=st.norm)


# looking on the graph we can conclude that SalePrice is not normally distributed.

# In[ ]:


print(train["SalePrice"].skew())
print(train["SalePrice"].kurt())


# Positive skewness indicates that SalePrice is skewed to right.

# 
# ### What effect does it have on our prediction if SalePrice is not normally distributed and is skewed?
# We will come to this point later. But now we should see what are the assumptions in Regression(Considering Ridge and lasso).
# 1. Dependent variable(SalePrice) should be approximately normally distributed. And henceforth should not be skewed.
# 2. residual error should be normally distributed. Plotting a graph of residual error vs our prediction gives us a idea about if residual is normally distributed and remedy steps we should take if it is not.
# 
# 

# ## Some numerical features are categorical. So we would type convert them to categorical.
# Let's plot there scatter plot to see who are those numeric features.

# In[ ]:


def pairplot(x, y, **kwargs):
    ax = plt.gca()
    ts = pd.DataFrame({'time': x, 'val': y})
    ts.plot.scatter('time', 'val',ax=ax)
f = pd.melt(train, id_vars=["SalePrice"], value_vars=numeric_features[:-1])
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False,size=5 )
g = g.map(pairplot, "value", "SalePrice")


# In[ ]:


from IPython.display import display, HTML
pd.options.display.max_rows = 10
pd.options.display.max_columns = 90
display(train)



Numerical features: OverallQual, OverallCond, MoSold, MSSubClass. are more like categorical feature so let's convert them to categorical feature.
But OverallQual, OverallCond are ordinal variable So we will leave them as they are.

# In[ ]:


#Converting MoSold,MSSubClass to categorical feature
num_to_cat = ['MSSubClass', "MoSold"]
for col in num_to_cat:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')
    numeric_features.remove(col)
    category_feature.append(col)


# ## Correlation between variables
#   We will see the corealation between numeric features and SalePrice

# In[ ]:


pd.options.display.max_rows = 45
pd.options.display.max_columns = 45
corrmat = train.corr()
corrmat.sort_values("SalePrice",inplace = True, ascending= False)
# print(sorted(list(corrmat["SalePrice"].values)))
plt.figure(figsize=(20,10))
sns.heatmap(corrmat, vmax=.8, square=True );
display(corrmat.shape)


# In[ ]:


# We will take top 10 most correlated components
#https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
k= 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=False, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# 
# 
#    1. TotRmsAbvGrd and GrLivArea are significantly correlated so we will only have GrLivArea(Pending FullBath)
#    2. GarageCars and GarageArea are highly corelated with each other. So we will only reatain GarageArea. Also GarageYrBlt is highy correlated to GarageArea but I won't drop it as what is the GarageArea have to take with GarageYrBlt also drop is.
#    3. TotalBsmtSF and 1stFlrSF are highly correlated so we will drop 1stFlrSF
#    4. Pending correlation between YearRemoteAdd and YearBuilt
# So now our top components having high corelation with SalePrice are: , GrLivArea, GarageCars, TotalBsmtSF, FullBath, YearBuilt, YearRemodAdd
# 

# In[ ]:


len(train.columns)


# In[ ]:


# droping TotRmsAbvGrd, 1stFlrSF,  and GarageCars
drop = ['3SsnPorch', 'MiscVal','PoolArea',"TotRmsAbvGrd", "1stFlrSF", "GarageCars","GarageQual",
        'ScreenPorch']
train.drop(drop, axis = 1, inplace = True)
test.drop(drop, axis = 1, inplace = True)
len(train.columns)
numeric_features = [i for i in numeric_features if i not in drop]
category_feature = [i for i in category_feature if i not in drop]


# ## Dropping Categorical Data with very low variance
# Idea is adopted from the
# notebook of PhilipBall, Link to the notebook is given here
# [Notebook](https://www.kaggle.com/fiorenza2/journey-to-the-top-10)

# In[ ]:


cat_lowVar = [col for col in category_feature if sum(train[col] == train[col].mode()[0])/len(train) > 0.94]
cat_lowVar


# In[ ]:


train.drop(cat_lowVar, axis = 1, inplace = True)
test.drop(cat_lowVar, axis = 1, inplace = True)
category_feature = [i for i in category_feature if i not in cat_lowVar]


# ## Removing Outliars(numerical features)
#    **Numerical feature having outliars are**:
#              1. GrLivArea -> 4
#              2. TotalBsmtSF -> 1
#              3. LotFrontage ->2
#              4. MasVnrArea ->1
#              5. BsmtFinSF1 ->1

# In[ ]:


#Removing OutLiars:
#GrLivArea
temp = train["GrLivArea"]>4000
train.drop((temp[temp == True].index), axis = 0, inplace = True)
#TotalBsmtSF
temp = train["TotalBsmtSF"]>4000
train.drop((temp[temp == True].index), axis = 0, inplace = True)
#LotFrontage
temp = train["LotFrontage"]>300
train.drop((temp[temp == True].index), axis = 0, inplace = True)
#MasVnrArea
temp = train["MasVnrArea"]>1500
train.drop((temp[temp == True].index), axis = 0, inplace = True)
#BsmtFinSF
temp = train["BsmtFinSF1"]>5000
train.drop((temp[temp == True].index), axis = 0, inplace = True)


# In[ ]:


train.drop(train.index[train['SalePrice']>500000], axis = 0, inplace = True)


# In[ ]:


print(297 in train.index)


# ## Box-Plot some numerical features which have discrete values

# In[ ]:


# box_feature= ["YrSold", "Fireplaces", "KitchenAbvGr", "FullBath", "HalfBath", "BsmtFullBath",
#              "BsmtHalfBath", "MSSubClass","OverallQual", "OverallCond", "MoSold"]
# train_temp = train.copy()
# for col in box_feature:
#     train[col] = train[col].astype(str)
# def boxplot(x, y, **kwargs):
#     sns.boxplot(x=x, y=y)
#     x=plt.xticks(rotation=90)
# f = pd.melt(train_temp, id_vars=['SalePrice'], value_vars=box_feature)
# g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
# g = g.map(boxplot, "value", "SalePrice")

# del train_temp


# In[ ]:


#print numeric and categorical feature again
numeric_features_num = len(numeric_features)
category_feature_num = len(category_feature)
print("No of numeric Features: ",numeric_features_num,"\n", numeric_features)
print("No of Categorical Features: ",category_feature_num,"\n", category_feature)


# ## Box Plot of categorical data

# In[ ]:


# for col in box_feature:
#     train[col] = train[col].astype(str)
def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=90)
f = pd.melt(train, id_vars=['SalePrice'], value_vars=category_feature)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(boxplot, "value", "SalePrice")


# ### REMOVE OUTLIERS IN CATEGORICAL DATA

# In[ ]:


#MSZONING

#Alley
display(train.shape)
temp = train["Alley"].notnull()
temp = temp[temp == True].index
temp = train.loc[temp]["SalePrice"]>250000
train.drop((temp[temp == True].index), axis = 0, inplace = True)
display(train.shape)


# 
# ## Missing Data
# ### 1. Numerical

# In[ ]:


print("Training Data: ",train.shape)
missing_num = train[numeric_features].isnull().sum()
missing_num = missing_num[missing_num>0]
missing_num.sort_values(inplace = True, ascending = False)
missing_num = pd.DataFrame({"missing":missing_num})
display(missing_num)
missing_num1 = test[numeric_features].isnull().sum()
missing_num1 = missing_num1[missing_num1>0]
missing_num1.sort_values(inplace=True,ascending=False)
missing_num1 = pd.DataFrame({"missing": missing_num1})
print("Test Data: ",test.shape)
display(missing_num1)


# ### Train Data: Replace Nan in Lot Frontage and MasVnrArea with their mean value GarageYrBlt with mode.
# ### Test Data: Replace missing data with their mean, except in 'BsmtHalfBath' and 'BsmtFullBath', these have discrete value hence replace with mode().
#             

# In[ ]:


pd.options.display.max_rows = 1400
pd.options.display.max_columns = 70
train.fillna({'LotFrontage':train['LotFrontage'].mean(),
              'MasVnrArea':train['MasVnrArea'].mean(), 'GarageYrBlt':
             train['GarageYrBlt'].mode().values[0]}, inplace = True )
test.fillna({'BsmtHalfBath':test['BsmtHalfBath'].mode()[0], 
             'BsmtFullBath':test['BsmtFullBath'].mode()[0],'GarageYrBlt':
             test['GarageYrBlt'].mode()[0]},inplace = True)
test.fillna(test[missing_num1.index].mean(), inplace = True)

print(test[numeric_features[:-1]].isnull().any().any())
print(train[numeric_features[:-1]].isnull().any().any())


# ### 2. Categorical
# 

# In[ ]:


missing_cat = train[category_feature].isnull().sum()
missing_cat = missing_cat[missing_cat>0]
missing_cat.sort_values(inplace=True,ascending=False)
missing_cat = pd.DataFrame({"missing": missing_cat})
print("Training Data: ")
display(missing_cat)
# train[missing_cat.index] = train[missing_cat.index].astype('category')
for col in category_feature:
    train[col] = train[col].astype('category')
# print(train[missing.index])
missing_cat1 = test[category_feature[:-1]].isnull().sum()
missing_cat1 = missing_cat1[missing_cat1>0]
missing_cat1.sort_values(inplace=True,ascending=False)
missing_cat1 = pd.DataFrame({"missing": missing_cat1})
print("\nTest Data: ")
display(missing_cat1)

for col in category_feature:
    test[col] = test[col].astype('category')
# missing_electrical = train[train["Electrical"].isnull()].index.tolist()
# print(missing_electrical)


# 1. GarageCond, GarageQual, Garagefinish, GarageType should be filled as none because whenever value is none corresponding GarageArea is 0 means no garage is available
# 2. BsmtCond, BsmtQual, BsmtExposure, BsmtFinType2, BsmtFinType1
# - BsmtExposure NaN value should be filled with "N0"
# 3. NaN value of fence means no fence
# 4. FireplaceQU, NaN value should be filled with none because corresponding number of places 
# 5. MasVnrType should be filled none because corresponding MasvnrArea is 0
# 6. I don't have any idea about electrical. So I will remove the row containing missing data

# In[ ]:


train['BsmtExposure'].fillna('No', inplace = True)
test['BsmtExposure'].fillna('No', inplace = True)
print(train['BsmtExposure'].isnull().any().any())
for col in missing_cat.index:
    train[col].cat.add_categories("none", inplace = True)
    train[col].fillna("none", inplace = True)
train.drop(train.index[1379], inplace = True)

train['BsmtExposure'].cat.remove_unused_categories(inplace = True)  #"none" is unused category
print(train['BsmtExposure'].isnull().any().any())
#Filling missing data in test
temp = list(missing_cat.index)
temp = temp[:-1]
for col in temp:
    test[col].cat.add_categories("none", inplace = True)
    test[col].fillna("none", inplace = True)
test['BsmtExposure'].cat.remove_unused_categories(inplace = True)
df =test[missing_cat1.index].mode()
df1 = pd.Series(list(df.values[0,:]), index = df.columns)
test.fillna(df1, inplace = True)


print(test.isnull().any().any())
print(train.isnull().any().any())


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


col_skew = (train[numeric_features].skew()[train.skew()>2]).index
col_skew


# In[ ]:


# display(test['MasVnrArea'].min())
# for col in col_skew:
#     train[col] = np.log1p(train[col])
#     test[col] = np.log1p(test[col])


# ## Encoding Categorical Variable
# ### 1.Training Data
# ** Fisr Let's print various categories and divide them in nominal and ordinal variable**

# In[ ]:


for col in category_feature:
    print(col)
    print(train[col].cat.categories)


# **Ordinal Variables**: PoolQC, FireplaceQu, GarageCond, GarageQual, BsmtExposure, BsmtCond, BsmtQual, ExterQual,
#                                       ExterCond, HeatingQC, KitchenQual, *MSSubClass, MoSold*

# In[ ]:


ord_cat = ["PoolQC", "FireplaceQu", 'BsmtCond', 'GarageCond', 
           'BsmtQual', 'ExterQual','ExterCond', 'HeatingQC', 'KitchenQual']
dic_num = {'none': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
#, 'GarageCond', 'GarageQual'


# In[ ]:


#BsmtExposure has categories: Av, Gd, Mn, No 
# train['BsmtExposure'].replace({'No':0,'Av':1,'Gd':2,'Mn':3},inplace = True)
cat1 = list(train['MSSubClass'].cat.categories)
d1 = dict(zip(cat1, np.arange(1,len(cat1)+1)))
print(d1)
train.replace({'MSSubClass': d1}, inplace = True)
for col in ord_cat:
    train[col].replace(dic_num, inplace = True)
    train[col] = train[col].astype('float64')


# In[ ]:


pd.options.display.max_columns = 70
print((train.select_dtypes(include= ['category']).columns))


# ## 2. Test Data(Encoding Categorical Variable)

# In[ ]:


for col in category_feature:
    print(col)
    print(test[col].cat.categories)


# **Ordinal Variable**: ExterQual, ExterCond, BsmtQual, BsmtExposure, BsmtCond,  HeatingQC, FireplaceQu, GarageQual, GarageCond, PoolQC, MSSubClass, MoSold

# In[ ]:


ord_cat1 = ['ExterQual', 'ExterCond', 'BsmtQual', 'GarageCond', 'BsmtCond', 'HeatingQC', 
           'FireplaceQu', 'PoolQC','KitchenQual']
#'GarageQual', 'GarageCond'


# In[ ]:


#BsmtExposure has categories: Av, Gd, Mn, No 
# print(test['BsmtExposure'].head())
# test['BsmtExposure'].replace({'No':0,'Av':1,'Gd':2,'Mn':3}, inplace = True)
# test['BsmtExposure'] = test['BsmtExposure'].astype('float64')
cat1 = list(test['MSSubClass'].cat.categories)
d1 = dict(zip(cat1, np.arange(1,len(cat1)+1)))
test.replace({'MSSubClass': d1}, inplace = True)
for col in ord_cat1:
    test[col].replace(dic_num, inplace = True)
    test[col] = test[col].astype('float64')
# print(test['BsmtExposure'].head())


# In[ ]:


numeric_features = test.select_dtypes(exclude=['O','category']).columns
# print(numeric_features)
display(train['ExterQual'].head())
test['ExterQual'] = test['ExterQual'].astype('float64')
def pairplot(x, y, **kwargs):
    ax = plt.gca()
    ts = pd.DataFrame({'time': x, 'val': y})
#     print(val)
    ts.plot.scatter('time', 'val',ax=ax)
f = pd.melt(train, id_vars=["SalePrice"], value_vars=numeric_features[:-1])
# print(f)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False,size=5 )
g = g.map(pairplot, "value", "SalePrice")


# In[ ]:


pd.options.display.max_columns = 90
SP = train.pop('SalePrice')
# SP.index = np.arange(len(SP))
print(SP.isnull().any())
print(train.shape)
test.shape


# In[ ]:


all_data = pd.concat([train, test])
print(all_data.shape)
print(all_data['Alley'].cat.categories)
all_data['Alley'].replace({"none":0, "Grvl":1, "Pave":2}, inplace = True)
all_data['Alley'] = all_data["Alley"].astype('float64')
all_data["Alley_sq"] = all_data['Alley']**2
######Droping Columns and experimenting######
all_data.drop("PoolQC", axis = 1, inplace =True)
# all_data["GarageCond"] = all_data["GarageCond"].astype('category')
############################################
# all_data = all_data.sample(frac=1).reset_index(drop=True)
all_data = pd.get_dummies(all_data)
pd.options.display.max_columns = 300
# print(all_data.head())
all_data['OverallQual_sq'] = all_data['OverallQual']**2
all_data['OverallQual_cb'] = all_data['OverallQual']**3
all_data['OverallQualCond'] = all_data['OverallQual']*all_data['OverallCond']
all_data['TotalBsmtSF_cb'] = all_data['TotalBsmtSF']**2
all_data['GarageArea_cb'] = all_data['GarageArea']**3
all_data['KitchenQual_sq'] = all_data['KitchenQual']**2
all_data['KitchenQual_cb'] = all_data['KitchenQual']**3
all_data['FireplaceQu_sq'] = all_data['FireplaceQu']**2
all_data['BsmtQual_sq'] = all_data['BsmtQual']**2
all_data['BsmtQual_cb'] = all_data['BsmtQual']**3
# all_data['BsmtQualCond'] = all_data['BsmtQual']*all_data['BsmtCond']
all_data['GarageAreaCond'] = all_data['GarageArea']*all_data['GarageCond']
all_data['GrLivArea_sq'] = all_data['GrLivArea']**2
all_data['GrLivArea_cb'] = all_data['GrLivArea']**3
train1 = all_data.iloc[:1444,:]
train1['SalePrice'] = SP
# print(train1['SalePrice'].isnull().any())
# train1 = train1.sample(frac=1).reset_index(drop=True)
display(train1['SalePrice'])#
SP = train1.pop('SalePrice')
test1 = all_data.iloc[1444:,:]


# In[ ]:


display(train1.shape)
print(test1.shape)


# In[ ]:



x_test = train1.values[1022:,:]
y_test = SP.values[1022:]


# In[ ]:


import numpy as np
def rmsle(predicted, actual):
    assert(len(predicted) == len(actual))
    p = np.log(np.array(predicted) + 1)
    a = np.log(np.array(actual) + 1)
    return (((p - a)**2).sum() / len(predicted))**0.5

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
it = [200, 300, 400,500, 600, 700, 800, 900, 1000, 1022]
J1,J2 = [],[]
for iteration in it:
    lr = LinearRegression()
    lr.fit(train1.values[:iteration,:], SP.values[:iteration])
    print(iteration)
    score1 = lr.score(train1.values[:iteration,:], SP.values[:iteration])
    score2 = lr.score(x_test, y_test)
    J1.append(1-score1)
    J2.append(1-score2)
    print(1-score1, 1-score2)
    
plt.plot(J1,'b-',label="J_train")
plt.plot(J2,'r-',label="J_test")
# plt.label()
plt.show()
y_pred1 = lr.predict(x_test)
error = rmsle(y_pred1, y_test)
print(error)
y_pred = lr.predict(test1.values)
y_pred[y_pred<0] = 150000
sub1 = pd.DataFrame()
sub1['Id']=ID
sub1['SalePrice']=y_pred
sub1.to_csv("prediction_LRegr.csv",index=False)


# In[ ]:


(lr.coef_[lr.coef_<0]).shape


# In[ ]:


import numpy as np
def rmsle(predicted, actual):
    assert(len(predicted) == len(actual))
    p = np.log(np.array(predicted) + 1)
    a = np.log(np.array(actual) + 1)
    return (((p - a)**2).sum() / len(predicted))**0.5

from sklearn.linear_model import Lasso
lr =Lasso(0.3)
it = [200, 300, 400,500, 600, 700, 800, 900, 950,1000, 1022,1444]
J1,J2 = [],[]
# alpha =[1,10,100,140,200,1000]
alpha = [100]
for al in alpha:
    print(al)
    J1,J2= [],[]
    for iteration in it:
        lr = Lasso(al, max_iter= 10000)
        lr.fit(train1.values[:iteration,:], SP.values[:iteration])
        print(iteration)
        score1 = lr.score(train1.values[:iteration,:], SP.values[:iteration])
        score2 = lr.score(x_test, y_test)
#         y_pred1 = lr.predict(train1.values[:iteration,:])
#         score1 = rmsle(y_pred1, SP.values[:iteration])
#         y_pred1 = lr.predict(x_test)
#         score2 = rmsle(y_pred1, y_test)
        J1.append(1-score1)
        J2.append(1-score2)
        print(1-score1, 1-score2)
    
    plt.plot(J1,'b-',label="J_train")
    plt.plot(J2,'r-',label="J_test")
    # plt.label()
    plt.show()
    y_pred1 = lr.predict(x_test)
    error = rmsle(y_pred1, y_test)
    print(error)
y_pred = lr.predict(test1.values)
y_pred[y_pred<0] = 150000
sub1 = pd.DataFrame()
sub1['Id']=ID
sub1['SalePrice']=y_pred
sub1.to_csv("prediction_LRegr1.csv",index=False)


# In[ ]:



(lr.coef_[lr.coef_ < 0.01]).shape

