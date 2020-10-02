#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import numpy as np
import pandas as pd

# Visualization 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('whitegrid')

import warnings
warnings.simplefilter(action='ignore')

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score,GridSearchCV,KFold,RandomizedSearchCV,train_test_split
import math
import sklearn.model_selection as ms
import sklearn.metrics as sklm


# In[ ]:


# Exploratory data analysis
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


print('The shape of our training set: ',train.shape[0], 'houses', 'and', train.shape[1], 'features')
print('The shape of our testing set: ',test.shape[0], 'houses', 'and', test.shape[1], 'features')
print('The testing set has 1 feature less than the training set, which is SalePrice, the target to predict  ')


# In[ ]:


# Numerical value correaltion with sales Pricw i.we the target variable
num = train.select_dtypes(exclude='object')
numcorr = num.corr()
f,ax = plt.subplots(figsize=(17,1))
sns.heatmap(numcorr.sort_values(by=['SalePrice'],ascending=False).head(1),cmap='Blues')
plt.title('Numerical features correlation with the sales price', weight='bold',fontsize=18)
plt.xticks(weight = 'bold')
plt.yticks(weight='bold',color='dodgerblue',rotation=0)
plt.show()


# In[ ]:


Num = numcorr['SalePrice'].sort_values(ascending=False).head(10).to_frame()

cm = sns.light_palette('violet',as_cmap=True)
s = Num.style.background_gradient(cmap=cm)
s


# In[ ]:


plt.figure(figsize=(10,6))
plt.scatter(x=train['GrLivArea'],y=train['SalePrice'],color = 'blue',alpha=0.5)
plt.title('Ground Living area/ Sale Price',weight = 'bold',fontsize=16)
plt.xlabel('Ground living area',weight='bold',fontsize=12)
plt.ylabel('Sale price',weight = 'bold',fontsize=12)
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.show()


# In[ ]:


# Figure Size
fig, ax = plt.subplots(figsize=(10,6))
train['Neighborhood'].value_counts().sort_values(ascending=True).plot(kind='barh',color='red',alpha=0.5)
plt.xlabel('Count',weight='bold',fontsize=12)
plt.title('Most Frequent Neighborhoods',weight='bold',fontsize=14)

for i in ax.patches:
    ax.text(i.get_width()+1, i.get_y()+0.25, str(round((i.get_width()), 2)),
            fontsize=10, fontweight='bold', color='grey')
plt.yticks(weight='bold')
plt.show()


# In[ ]:


# Figure Size
fig, ax = plt.subplots(figsize=(10,6))
train['BldgType'].value_counts().sort_values(ascending=True).plot(kind='barh',color='green',alpha=0.5,ax=ax)
plt.xlabel('Count',weight='bold',fontsize=12)
plt.title('Building type: Type of dwelling',loc='center',weight='bold',fontsize=14)


for i in ax.patches:
    ax.text(i.get_width()+1, i.get_y()+0.25, str(round((i.get_width()), 2)),
            fontsize=10, fontweight='bold', color='grey',va="center")

plt.yticks(weight='bold')
plt.show()


# In[ ]:


na = train.shape[0]
nb = test.shape[0]
y_train = train['SalePrice'].to_frame()
#Combine train and test sets
c1 = pd.concat((train, test), sort=False).reset_index(drop=True)
#Drop the target "SalePrice" and Id columns
c1.drop(['SalePrice'], axis=1, inplace=True)
c1.drop(['Id'], axis=1, inplace=True)
print("Total size is :",c1.shape)


# In[ ]:


missing = pd.DataFrame()
t1 =(round((c1.isnull().sum()/len(c1)*100),2))
col_name = t1.index.tolist()
t1 = list(t1.sort_values(ascending=False)) 
missing['Col_Name']=col_name
missing['Pecentage']=t1
missing


# In[ ]:


def msv1(data, thresh=20, color='black', edgecolor='black', width=15, height=3):
    """
    SOURCE: https://www.kaggle.com/amiiiney/price-prediction-regularization-stacking
    """
    
    plt.figure(figsize=(width,height))
    percentage=(data.isnull().mean())*100
    percentage.sort_values(ascending=False).plot.bar(color=color, edgecolor=edgecolor)
    plt.axhline(y=thresh, color='r', linestyle='-')
    plt.title('Missing values percentage per column', fontsize=20, weight='bold' )
    plt.text(len(data.isnull().sum()/len(data))/1.7, thresh+12.5, 'Columns with more than %s%s missing values' %(thresh, '%'), fontsize=12, color='crimson',
         ha='left' ,va='top')
    plt.text(len(data.isnull().sum()/len(data))/1.7, thresh - 5, 'Columns with less than %s%s missing values' %(thresh, '%'), fontsize=12, color='green',
         ha='left' ,va='top')
    plt.xlabel('Columns', size=15, weight='bold')
    plt.ylabel('Missing values percentage')
    plt.yticks(weight ='bold')
    
    return plt.show()
msv1(c1, 20, color=('silver', 'gold', 'lightgreen', 'skyblue', 'lightpink'))


# In[ ]:


c=c1.dropna(thresh=len(c1)*0.8, axis=1)
print('We dropped ',c1.shape[1]-c.shape[1], ' features in the combined set')


# In[ ]:


allna = (c.isnull().sum() / len(c))*100
allna = allna.drop(allna[allna == 0].index).sort_values()

def msv2(data, width=12, height=8, color=('silver', 'gold','lightgreen','skyblue','lightpink'), edgecolor='black'):
    """
    SOURCE: https://www.kaggle.com/amiiiney/price-prediction-regularization-stacking
    """
    fig, ax = plt.subplots(figsize=(width, height))

    allna = (data.isnull().sum() / len(data))*100
    tightout= 0.008*max(allna)
    allna = allna.drop(allna[allna == 0].index).sort_values().reset_index()
    mn= ax.barh(allna.iloc[:,0], allna.iloc[:,1], color=color, edgecolor=edgecolor)
    ax.set_title('Missing values percentage per column', fontsize=15, weight='bold' )
    ax.set_xlabel('Percentage', weight='bold', size=15)
    ax.set_ylabel('Features with missing values', weight='bold')
    plt.yticks(weight='bold')
    plt.xticks(weight='bold')
    for i in ax.patches:
        ax.text(i.get_width()+ tightout, i.get_y()+0.1, str(round((i.get_width()), 2))+'%',
            fontsize=10, fontweight='bold', color='grey')
    return plt.show()


# In[ ]:


msv2(c)


# In[ ]:


print('The shape of the combined dataset after dropping features with more than 80% M.V.', c.shape)


# In[ ]:


NA=c[allna.index.to_list()]


# In[ ]:


NAcat=NA.select_dtypes(include='object')
NAnum=NA.select_dtypes(exclude='object')
print('We have :',NAcat.shape[1],'categorical features with missing values')
print('We have :',NAnum.shape[1],'numerical features with missing values')


# In[ ]:


# Numerical Feature
NAnum.head()


# In[ ]:


#MasVnrArea: Masonry veneer area in square feet, the missing data means no veneer so we fill with 0
c['MasVnrArea']=c.MasVnrArea.fillna(0)
#LotFrontage has 16% missing values. We fill with the median
c['LotFrontage']=c.LotFrontage.fillna(c.LotFrontage.median())
#GarageYrBlt:  Year garage was built, we fill the gaps with the median: 1980
c['GarageYrBlt']=c["GarageYrBlt"].fillna(1980)
#For the rest of the columns: Bathroom, half bathroom, basement related columns and garage related columns:
#We will fill with 0s because they just mean that the hosue doesn't have a basement, bathrooms or a garage


# In[ ]:


# 2.3 Categorical features:
# And we have 18 Categorical features with missing values:

# Some features have just 1 or 2 missing values, so we will just use the forward fill method because they are obviously values that can't be filled with 'None's
# Features with many missing values are mostly basement and garage related (same as in numerical features)
# so as we did with numerical features (filling them with 0s), we will fill the categorical missing values with "None"s assuming that the houses lack basements and garages.
NAcat.head()


# In[ ]:


# Number of missing percolumns
NAcat1= NAcat.isnull().sum().to_frame().sort_values(by=[0]).T
cm = sns.light_palette("lime", as_cmap=True)

NAcat1 = NAcat1.style.background_gradient(cmap=cm)
NAcat1


# In[ ]:


# The table above helps us to locate the categorical features with few missing values.

# We start our cleaning with the features having just few missing value (1 to 4): We fill the gap with forward fill method:

fill_cols = ['Electrical', 'SaleType', 'KitchenQual', 'Exterior1st',
             'Exterior2nd', 'Functional', 'Utilities', 'MSZoning']

for col in c[fill_cols]:
    c[col] = c[col].fillna(method='ffill')


# In[ ]:


#Categorical missing values
NAcols=c.columns
for col in NAcols:
    if c[col].dtype == "object":
        c[col] = c[col].fillna("None")
#Numerical missing values
for col in NAcols:
    if c[col].dtype != "object":
        c[col]= c[col].fillna(0)


# In[ ]:


c.isnull().sum().sort_values(ascending=False).head()


# In[ ]:


# 3- Feature engineering:
# Since the area is a very important variable, we will create a new feature "TotalArea" that sums the area of all the floors and the basement.

# Bathrooms: All the bathroom in the ground floor
# Year average: The average of the sum of the year the house was built and the year the house was remodeled
c['TotalArea'] = c['TotalBsmtSF'] + c['1stFlrSF'] + c['2ndFlrSF'] + c['GrLivArea'] +c['GarageArea']

c['Bathrooms'] = c['FullBath'] + c['HalfBath']*0.5 

c['Year average']= (c['YearRemodAdd']+c['YearBuilt'])/2


# In[ ]:


# 4- Encoding categorical features:
# 4.1 Numerical features:
# We start with numerical features that are actually categorical, for example "Month sold", the values are from 1 to 12, each number is assigned to a month November is number 11 while March is number 3. 11 is just the order of the months and not a given value, so we convert the "Month Sold" feature to categorical

#c['MoSold'] = c['MoSold'].astype(str)
c['MSSubClass'] = c['MSSubClass'].apply(str)
c['YrSold'] = c['YrSold'].astype(str)


# In[ ]:


# 4.2 One hot encoding:
cb=pd.get_dummies(c)
print("the shape of the original dataset",c.shape)
print("the shape of the encoded dataset",cb.shape)
print("We have ",cb.shape[1]- c.shape[1], 'new encoded features')


# In[ ]:


# We are done with the cleaning and feature engineering. Now, we split the combined dataset to the original train and test sets

Train = cb[:na]  #na is the number of rows of the original training set
Test = cb[na:] 


# In[ ]:


a = train
b = test


# In[ ]:


# 5- Outliers detection:
# 5.1 Outliers visualization:
# This part of the kernel will be a little bit messy. 
# I didn't want to deal with the outliers in the combined dataset to keep the shape of the original train and test datasets. 
# Dropping them would shift the location of the rows.

# If you know a better solution to this, I will be more than happy to read your recommandations.

# OK. So we go back to our original train dataset to visualize the important features / Sale price scatter plot to find outliers



fig = plt.figure(figsize=(15,15))
ax1 = plt.subplot2grid((3,2),(0,0))
plt.scatter(x=a['GrLivArea'], y=a['SalePrice'], color=('yellowgreen'), alpha=0.5)
plt.axvline(x=4600, color='r', linestyle='-')
plt.title('Ground living Area- Price scatter plot', fontsize=15, weight='bold' )

ax1 = plt.subplot2grid((3,2),(0,1))
plt.scatter(x=a['TotalBsmtSF'], y=a['SalePrice'], color=('red'),alpha=0.5)
plt.axvline(x=5900, color='r', linestyle='-')
plt.title('Basement Area - Price scatter plot', fontsize=15, weight='bold' )

ax1 = plt.subplot2grid((3,2),(1,0))
plt.scatter(x=a['1stFlrSF'], y=a['SalePrice'], color=('deepskyblue'),alpha=0.5)
plt.axvline(x=4000, color='r', linestyle='-')
plt.title('First floor Area - Price scatter plot', fontsize=15, weight='bold' )

ax1 = plt.subplot2grid((3,2),(1,1))
plt.scatter(x=a['MasVnrArea'], y=a['SalePrice'], color=('gold'),alpha=0.9)
plt.axvline(x=1500, color='r', linestyle='-')
plt.title('Masonry veneer Area - Price scatter plot', fontsize=15, weight='bold' )

ax1 = plt.subplot2grid((3,2),(2,0))
plt.scatter(x=a['GarageArea'], y=a['SalePrice'], color=('orchid'),alpha=0.5)
plt.axvline(x=1230, color='r', linestyle='-')
plt.title('Garage Area - Price scatter plot', fontsize=15, weight='bold' )

ax1 = plt.subplot2grid((3,2),(2,1))
plt.scatter(x=a['TotRmsAbvGrd'], y=a['SalePrice'], color=('tan'),alpha=0.9)
plt.axvline(x=13, color='r', linestyle='-')
plt.title('TotRmsAbvGrd - Price scatter plot', fontsize=15, weight='bold' )
plt.show()


# In[ ]:


# /The outliers are the points in the right that have a larger area or value but a very low sale price.
# We localize those points by sorting their respective columns

# Interesting! The outlier in "basement" and "first floor" features is the same as the first outlier in ground living area: 
# The outlier with index number 1298.
# 5.2 Outliers localization:
# We sort the columns containing the outliers shown in the graph, we will use the function head() to show the outliers: 
# head(number of outliers or dots shown in each plot)


# In[ ]:


a['GrLivArea'].sort_values(ascending=False).head(2)


# In[ ]:


a['TotalBsmtSF'].sort_values(ascending=False).head(1)


# In[ ]:


a['MasVnrArea'].sort_values(ascending=False).head(1)


# In[ ]:


a['1stFlrSF'].sort_values(ascending=False).head(1)


# In[ ]:


a['GarageArea'].sort_values(ascending=False).head(4)


# In[ ]:


a['TotRmsAbvGrd'].sort_values(ascending=False).head(1)


# In[ ]:


train=Train[(Train['GrLivArea'] < 4600) & (Train['MasVnrArea'] < 1500)]

print('We removed ',Train.shape[0]- train.shape[0],'outliers')


# In[ ]:


# We do the same thing with "SalePrice" column, we localize those outliers and make sure they are the right outliers to remove.

# They both have the same price range as the detected outliers. So, we can safely drop them.
target=a[['SalePrice']]
target.loc[1298]


# In[ ]:


target.loc[523]


# In[ ]:


#pos = [1298,523, 297, 581, 1190, 1061, 635, 197,1328, 495, 583, 313, 335, 249, 706]
pos = [1298,523, 297]
target.drop(target.index[pos], inplace=True)


# In[ ]:


# P.S. I didn't drop all the outliers because dropping all of them led to a worst RMSE score. More investigation is needed to filter those outliers.

print('We make sure that both train and target sets have the same row number after removing the outliers:')
print( 'Train: ',train.shape[0], 'rows')
print('Target:', target.shape[0],'rows')


# In[ ]:


plt.style.use('seaborn')
sns.set_style('whitegrid')
fig = plt.figure(figsize=(15,5))
#1 rows 2 cols
#first row, first col
ax1 = plt.subplot2grid((1,2),(0,0))
plt.scatter(x=a['GrLivArea'], y=a['SalePrice'], color=('orchid'), alpha=0.5)
plt.title('Area-Price plot with outliers',weight='bold', fontsize=18)
plt.axvline(x=4600, color='r', linestyle='-')
#first row sec col
ax1 = plt.subplot2grid((1,2),(0,1))
plt.scatter(x=train['GrLivArea'], y=target['SalePrice'], color='navy', alpha=0.5)
plt.axvline(x=4600, color='r', linestyle='-')
plt.title('Area-Price plot without outliers',weight='bold', fontsize=18)
plt.show()


# Log transform skewed numeric features:
# We want our skewness value to be around 0 and kurtosis less than 3. 
# For more information about skewness and kurtosis,I recommend reading this article.
# 
# Here are two examples of skewed features: Ground living area and 1st floor SF. We will apply np.log1p to the skewed variables.

# In[ ]:


print("Skewness before log transform: ", a['GrLivArea'].skew())
print("Kurtosis before log transform: ", a['GrLivArea'].kurt())


# In[ ]:


from scipy.stats import skew

print("Skewness after log transform: ", train['GrLivArea'].skew())
print("Kurtosis after log transform: ", train['GrLivArea'].kurt())


# In[ ]:


plt.style.use('seaborn')
sns.set_style('whitegrid')
fig = plt.figure(figsize=(15,10))
#1 rows 2 cols
#first row, first col
ax1 = plt.subplot2grid((2,2),(0,0))
sns.distplot(a.GrLivArea, color='plum')
plt.title('Before: Distribution of GrLivArea',weight='bold', fontsize=18)
#first row sec col
ax1 = plt.subplot2grid((2,2),(0,1))
sns.distplot(a['1stFlrSF'], color='tan')
plt.title('Before: Distribution of 1stFlrSF',weight='bold', fontsize=18)


ax1 = plt.subplot2grid((2,2),(1,0))
sns.distplot(train.GrLivArea, color='plum')
plt.title('After: Distribution of GrLivArea',weight='bold', fontsize=18)
#first row sec col
ax1 = plt.subplot2grid((2,2),(1,1))
sns.distplot(train['1stFlrSF'], color='tan')
plt.title('After: Distribution of 1stFlrSF',weight='bold', fontsize=18)
plt.show()


# In[ ]:


print("Skewness before log transform: ", target['SalePrice'].skew())
print("Kurtosis before log transform: ",target['SalePrice'].kurt())


# In[ ]:


#log transform the target:
target["SalePrice"] = np.log1p(target["SalePrice"])


# In[ ]:


plt.style.use('seaborn')
sns.set_style('whitegrid')
fig = plt.figure(figsize=(15,5))
#1 rows 2 cols
#first row, first col
ax1 = plt.subplot2grid((1,2),(0,0))
plt.hist(a.SalePrice, bins=10, color='mediumpurple',alpha=0.5)
plt.title('Sale price distribution before normalization',weight='bold', fontsize=18)
#first row sec col
ax1 = plt.subplot2grid((1,2),(0,1))
plt.hist(target.SalePrice, bins=10, color='darkcyan',alpha=0.5)
plt.title('Sale price distribution after normalization',weight='bold', fontsize=18)
plt.show()


# The skewness and kurtosis values look fine after log transform. We can now move forward to Machine Learning.
# 
# P.S.To get our original SalePrice values back, we will apply np.expm1 at the end of the study to cancel the log1p transformation after training and testing the models.

# 6- Machine Learning:
# 6.1 Preprocessing
# We start machine learning by setting the features and target:
# 
# Features: x
# Target: y

# In[ ]:


x=train
y=np.array(target)


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = .3, random_state=0)


# We use RobustScaler to scale our data because it's powerful against outliers, we already detected some but there must be some other outliers out there, I will try to find them in future versions of the kernel

# In[ ]:


from sklearn.preprocessing import RobustScaler
scaler= RobustScaler()
# transform "x_train"
x_train = scaler.fit_transform(x_train)
# transform "x_test"
x_test = scaler.transform(x_test)
#Transform the test set
X_test= scaler.transform(Test)


# We first start by trying the very basic regression model: Linear regression.
# 
# We use 5- Fold cross validation for a better error estimate:
# 6.2 Linear regression

# 6.3 Regularization:
# Ridge regression:
# Minimize squared error + a term alpha that penalizes the error
# We need to find a value of alpha that minimizes the train and test error (avoid overfitting)

# In[ ]:


import sklearn.model_selection as GridSearchCV
from sklearn.linear_model import Ridge

ridge=Ridge()
parameters= {'alpha':[x for x in range(1,101)]}

ridge_reg=ms.GridSearchCV(ridge, param_grid=parameters, scoring='neg_mean_squared_error', cv=15)
ridge_reg.fit(x_train,y_train)
print("The best value of Alpha is: ",ridge_reg.best_params_)
print("The best score achieved with Alpha=11 is: ",math.sqrt(-ridge_reg.best_score_))
ridge_pred=math.sqrt(-ridge_reg.best_score_)


# In[ ]:


ridge_mod=Ridge(alpha=15)
ridge_mod.fit(x_train,y_train)
y_pred_train=ridge_mod.predict(x_train)
y_pred_test=ridge_mod.predict(x_test)

print('Root Mean Square Error train = ' + str(math.sqrt(sklm.mean_squared_error(y_train, y_pred_train))))
print('Root Mean Square Error test = ' + str(math.sqrt(sklm.mean_squared_error(y_test, y_pred_test))))   


# Next we try Lasso regularization: Similar procedure as ridge regularization but Lasso tends to have a lot of 0 entries in it and just few nonzeros (easy selection). In other words, lasso drops the uninformative features and keeps just the important ones.
# As with Ridge regularization, we need to find the alpha parameter that penalizes the error

# In[ ]:


# lasso Regression
from sklearn.linear_model import Lasso

parameters= {'alpha':[0.0001,0.0009,0.001,0.002,0.003,0.01,0.1,1,10,100]}


lasso=Lasso()
lasso_reg=ms.GridSearchCV(lasso, param_grid=parameters, scoring='neg_mean_squared_error', cv=15)
lasso_reg.fit(x_train,y_train)

print('The best value of Alpha is: ',lasso_reg.best_params_)


# In[ ]:


lasso_mod=Lasso(alpha=0.0009)
lasso_mod.fit(x_train,y_train)
y_lasso_train=lasso_mod.predict(x_train)
y_lasso_test=lasso_mod.predict(x_test)

print('Root Mean Square Error train = ' + str(math.sqrt(sklm.mean_squared_error(y_train, y_lasso_train))))
print('Root Mean Square Error test = ' + str(math.sqrt(sklm.mean_squared_error(y_test, y_lasso_test))))


# We check next, the important features that our model used to make predictions
# The number of uninformative features that were dropped. Lasso give a 0 coefficient to the useless features, we will use the coefficient given to the important feature to plot the graph

# In[ ]:


coefs = pd.Series(lasso_mod.coef_, index = x.columns)

imp_coefs = pd.concat([coefs.sort_values().head(10),
                     coefs.sort_values().tail(10)])
imp_coefs.plot(kind = "barh", color='yellowgreen')
plt.xlabel("Lasso coefficient", weight='bold')
plt.title("Feature importance in the Lasso Model", weight='bold')
plt.show()


# In[ ]:


print("Lasso kept ",sum(coefs != 0), "important features and dropped the other ", sum(coefs == 0)," features")


# In[ ]:


from sklearn.linear_model import ElasticNetCV

alphas = [0.000542555]
l1ratio = [0.1, 0.3,0.5, 0.9, 0.95, 0.99, 1]

elastic_cv = ElasticNetCV(cv=5, max_iter=1e7, alphas=alphas,  l1_ratio=l1ratio)

elasticmod = elastic_cv.fit(x_train, y_train.ravel())
ela_pred=elasticmod.predict(x_test)
print('Root Mean Square Error test = ' + str(math.sqrt(sklm.mean_squared_error(y_test, ela_pred))))
print(elastic_cv.alpha_)
print(elastic_cv.l1_ratio_)


# In[ ]:


def regularization(x,y,modelo=Ridge, scaler=RobustScaler):
    """"
    Function to automate regression with regularization techniques.
    x expects the features
    y expects the target
    modelo: Ridge(default), Lasso, ElasticNetCV
    scaler: RobustScaler(default), MinMaxSclaer, StandardScaler
    SOURCE: https://www.kaggle.com/amiiiney/price-prediction-regularization-stacking
    Contact: amineyamlahi@gmail.com
    """
    #Split the data to train/test
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = .3, random_state=0)
    
    #Scale the data. RobustSclaer default
    from sklearn.preprocessing import RobustScaler
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import StandardScaler
    
    scaler= scaler()
    # transform "x_train"
    x_train = scaler.fit_transform(x_train)
    # transform "x_test"
    x_test = scaler.transform(x_test)
    #Transform the test set
    X_test= scaler.transform(Test)
    
    if modelo != ElasticNetCV:
        if modelo == Ridge:
            parameters= {'alpha':[x for x in range(1,101)]}
        elif modelo == Lasso:
            parameters= {'alpha':[0.0001,0.0009,0.001,0.002,0.003,0.01,0.1,1,10,100]}
            
        model=modelo()
            
        model=ms.GridSearchCV(model, param_grid=parameters, scoring='neg_mean_squared_error', cv=15)
        model.fit(x_train,y_train)
        y_pred= model.predict(x_test)

        #print("The best value of Alpha is: ",model.best_params_)
        print("The best RMSE score achieved with %s is: %s " %(model.best_params_,
                  str(math.sqrt(sklm.mean_squared_error(y_test, y_pred)))))
    elif modelo == ElasticNetCV:
        alphas = [0.000542555]
        l1ratio = [0.1, 0.3,0.5, 0.9, 0.95, 0.99, 1]

        elastic_cv = ElasticNetCV(cv=5, max_iter=1e7, alphas=alphas,  l1_ratio=l1ratio)

        elasticmod = elastic_cv.fit(x_train, y_train.ravel())
        ela_pred=elasticmod.predict(x_test)
        print("The best RMSE score achieved with alpha %s and l1_ratio %s is: %s "
              %(elastic_cv.alpha_,elastic_cv.l1_ratio_,
            str(math.sqrt(sklm.mean_squared_error(y_test, ela_pred)))))
        
            


# In[ ]:


regularization(x,y,Ridge)


# In[ ]:


regularization(x,y, Lasso)


# In[ ]:


regularization(x,y, ElasticNetCV)


# In[ ]:


from xgboost.sklearn import XGBRegressor


# In[ ]:


xgb= XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.5, gamma=0,
             importance_type='gain', learning_rate=0.01, max_delta_step=0,
             max_depth=3, min_child_weight=0, missing=None, n_estimators=4000,
             n_jobs=1, nthread=None, objective='reg:squarederror', random_state=0,
             reg_alpha=0.0001, reg_lambda=0.01, scale_pos_weight=1, seed=None,
             silent=None, subsample=1, verbosity=1)
xgmod=xgb.fit(x_train,y_train)
xg_pred=xgmod.predict(x_test)
print('Root Mean Square Error test = ' + str(math.sqrt(sklm.mean_squared_error(y_test, xg_pred))))


# 6.5 ENSEMBLE METHODS:
# VOTING REGRESSOR:
# A voting regressor is an ensemble meta-estimator that fits base regressors each on the whole dataset. It, then, averages the individual predictions to form a final prediction.
# After running the regressors, we com

# In[ ]:


from sklearn.ensemble import VotingRegressor

vote_mod = VotingRegressor([('Ridge', ridge_mod), ('Lasso', lasso_mod), ('Elastic', elastic_cv), 
                            ('XGBRegressor', xgb)])
vote= vote_mod.fit(x_train, y_train.ravel())
vote_pred=vote.predict(x_test)

print('Root Mean Square Error test = ' + str(math.sqrt(sklm.mean_squared_error(y_test, vote_pred))))


# In[ ]:


# STACKING REGRESSOR:
# We stack all the previous models, including the votingregressor with XGBoost as the meta regressor:

from mlxtend.regressor import StackingRegressor


stregr = StackingRegressor(regressors=[elastic_cv,ridge_mod, lasso_mod, vote_mod], 
                           meta_regressor=xgb, use_features_in_secondary=True
                          )

stack_mod=stregr.fit(x_train, y_train.ravel())
stacking_pred=stack_mod.predict(x_test)

print('Root Mean Square Error test = ' + str(math.sqrt(sklm.mean_squared_error(y_test, stacking_pred))))


# In[ ]:


# Last thing to do is average our regressors and fit them on the testing dataset
# Averaging Regressors
final_test=(0.3*vote_pred+0.5*stacking_pred+ 0.2*y_lasso_test)
print('Root Mean Square Error test = ' + str(math.sqrt(sklm.mean_squared_error(y_test, final_test))))


# In[ ]:


# 6.6 Fit the model on test data
# Now, we fit the models on the test data and then submit it to the competition

# We apply np.expm1 to cancel the np.logp1 (we did previously in data processing) and convert the numbers to their original form
#VotingRegressor to predict the final Test
vote_test = vote_mod.predict(X_test)
final1=np.expm1(vote_test)

#StackingRegressor to predict the final Test
stack_test = stregr.predict(X_test)
final2=np.expm1(stack_test)

#LassoRegressor to predict the final Test
lasso_test = lasso_mod.predict(X_test)
final3=np.expm1(lasso_test)


# In[ ]:


#Submission of the results predicted by the average of Voting/Stacking/Lasso
final=(0.2*final1+0.6*final2+0.2*final3)

final_submission = pd.DataFrame({
        "Id": b["Id"],
        "SalePrice": final
    })
final_submission.to_csv("final_submission.csv", index=False)
final_submission.head()


# Help from this kernel: https://www.kaggle.com/amiiiney/price-prediction-regularization-stacking please upvote this kernel

# In[ ]:




