#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# <h1><b>Competition Description<h1>
#     
# ![image.png](https://storage.googleapis.com/kaggle-competitions/kaggle/5407/media/housesbanner.png)
# 

# <p>Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.
# 
# With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.<p>

# # Importing Libraries

# In[ ]:


#Importing python libraries
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from pandas import get_dummies
import matplotlib as mpl
import xgboost as xgb
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sklearn
import scipy
import numpy
import json
import sys
import csv
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn import linear_model


# In[ ]:


from scipy.stats import skew, norm, probplot, boxcox


# In[ ]:


# import train and test to play with it
train_df = pd.read_csv('../input/train.csv')
train_df_dup = train_df.copy()
test_df = pd.read_csv('../input/test.csv')
labels_df = train_df.pop('SalePrice') 

pd.set_option('display.max_rows',100)# amount of rows that can be seen at a time


# In[ ]:


labels_df.describe()


# In[ ]:





# ***Before preprocessing and after starting we must explore data using pandas functions and know about various minute details which must later be inspected.This step is crucial as we don't need to directly rush the process thus making some wrong assumptions and making our data inefficient. This is basically called EDA(Exploratory Data Analysis)***

# # Exploratory Data Analysis

# In[ ]:


train_df.shape,test_df.shape


# In[ ]:


data = pd.concat([train_df, test_df], keys=['train_df', 'test_df'])
print(data.columns) # check column decorations
print('rows:', data.shape[0], ', columns:', data.shape[1]) # count rows of total dataset
print('rows in train dataset:', train_df.shape[0])
print('rows in test dataset:', test_df.shape[0])


# In[ ]:


data.info()


# In[ ]:


nans = pd.concat([train_df.isnull().sum(), train_df.isnull().sum() / train_df.shape[0], 
                  test_df.isnull().sum(), test_df.isnull().sum() / test_df.shape[0]], axis=1, 
                 keys=['Train', 'Percentage', 'Test', 'Percentage'])
print(nans[nans.sum(axis=1) > 0])


# ***We have most null values in 'LotFrontage','Alley','FireplaceQu','Fence' and 'MiscFeature'.***
# <br>
# *Other features only have a few missing values.*

# In[ ]:


labels_df.describe()


# While going through various kernels ,I got this good reusable function made by Marcelo Marques in his kernel which is quite similar to function str function in R which returns the types, counts, distinct, count nulls, missing ratio and uniques values of each field/feature.
# 
# <br>
# This is the link to his kernel https://www.kaggle.com/mgmarques/houses-prices-complete-solution

# In[ ]:



def rstr(df, pred): 
    obs = df.shape[0]
    types = df.dtypes
    counts = df.apply(lambda x: x.count())
    uniques = df.apply(lambda x: [x.unique()])
    nulls = df.apply(lambda x: x.isnull().sum())
    distincts = df.apply(lambda x: x.unique().shape[0])
    missing_ration = (df.isnull().sum()/ obs) * 100
    skewness = df.skew()
    kurtosis = df.kurt() 
    print('Data shape:', df.shape)
    
    if pred:
        corr = df.corr()[pred]
        str = pd.concat([types, counts, distincts, nulls, missing_ration, uniques, skewness, kurtosis, corr], axis = 1, sort=False)
        corr_col = 'corr '  + pred
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing_ration', 'uniques', 'skewness', 'kurtosis', corr_col ]
    
    str.columns = cols
    dtypes = str.types.value_counts()
    print('___________________________\nData types:\n',str.types.value_counts())
    print('___________________________')
    return str


# In[ ]:


details = rstr(train_df_dup, 'SalePrice')
display(details.sort_values(by='corr SalePrice', ascending=False))


# In[ ]:


train_df_dup.corr()['SalePrice']


# In[ ]:


test_df.head()


# In[ ]:


test_df.isnull().sum()


# In[ ]:


# Now its time for insightful visualization


# # Visualisation

# * First we will understand the relationship of most important varible with 'SalePrice'. 
# * For Visualisation purpose I use Tableau Software with easy to visualise-use-create data.
# * We will analyse bivariate relationship of each data with 'SalePrice' as one variable. Also we will use max.correlation as a measure.
# 

# **Overall Quaity vs Saleprice**

# ![Overall Quality vs Saleprice](https://storage.googleapis.com/kagglesdsdata/datasets/213949/465252/Qual%20vs%20Price.PNG?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1559899756&Signature=Gd9lVEPuFABK7TyzmQl6n1NkhzdVSyfa3eOx8FldAonVWAEQC6BREF2HJbHPiisElrZzwV94XLyx8ajdjkoq6QUeKP2BgsGeCQPmR8XE2BdnMEtZ7muEc7B%2F8fMksVM5a67r3wjhnlkJ2lF4WkblyXzVEY5Ry5b5bUAINHlhESJWB589PGVKvfvznKPssImj2PjVo6G3rFUATXYzuUN4wLU5KT9rVTy8TnZEK6CXqFRHCrtGdgCz1W4%2FuqdfpK0%2F1c4kLLww32Oj3snZgfg2Cbh2ryyz7Z3%2FZ2tOe35nIVudy0vrw9wCE0bn56VpWIpdHAwVBeJ9MXQpkg40VZaLLg%3D%3D)

# In[ ]:


# We can clearly see a clear and strong relationship between the 'SalePrice' and 'OverQual' which makes perfect sense as we all
# know that ovelall quality of house matters the most while buying the house.


# **GrlivArea vs Med.SalePrice**

# ![Ground-Living-Area vs SalesPrice](https://storage.googleapis.com/kagglesdsdata/datasets/216698/469917/GrlivArea%20vs%20Sale.PNG?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1559902333&Signature=eGr0DfXQWH825dbL8QG0PbElbzb1w7hSHMOG%2BrCKdDzH89v1svw%2FPaR%2F%2BjM026pdP5LcsZYtmnyEv9mvGRkTz7mUjtgBIiztv2QFw0QhGjmA4M7%2B1uBL68982XW%2BieUPuOk8RXRpSWZ8Tq%2FHYylFHkSktKxgBO5yt7vztp4DdM%2FkWphji%2BmL2bNTigH1amxpx%2BjMqDNEYJ6OXUFGc0%2BwJkhSB1mcfY7bonkPXmS6AaMiDXdHQTbOy9NARLaN0mY3sK5PUYFE8NvPvPEAqeY3AhUbfh4zE%2FHrcIag7I0c3FE6MMqszjGg7ECqBgH%2F8t1dpJh65FzfdokvXMG1bsygAQ%3D%3D)

# There is somewhat a positive correleation between the sale price and Living Area as the size of living area is priority among most people
# because of family size and other reasons. But as the end there is sharp downfall of price maybe other factors played a more important role at that place. Well that last part can be considered as a possible outliers.

# **Garage Size(in cars) vs Saleprice**

# ![](http://)

# ![**Garage Size(in cars) vs Saleprice**](https://storage.googleapis.com/kagglesdsdata/datasets/216698/469917/Garagecars.PNG?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1559902325&Signature=mecgfn1IjMguVsSWmuXQNISTiCvHHF63IGssqdD7R%2FS9FPZ4ARVMMnHzQ2QYr%2B6XPA%2F7%2FzbAB0wEN%2FABLBqb7wJK0tkP9iGuhA%2FZE7Xx0ksHgm7CwdYb%2FIqTxih3Pl%2Fyh1TSzkM0XF9PXrKr0BNPVp%2BYI5jWi6CY59%2Fs%2BLQ25%2BvF%2BQOD0OJRXwU%2FvTXcsyKHe4y41QaRyt6AV4FAGLuPYHaD585t3tTEeFQZP%2BfAQoO%2FRbl0dll1UUpLPz2BrZlMFq%2F8XSa14OtwxgjXI5USmpNzdCFM%2FPwTzoQ8xltY2JAVG2UV5NpAZY004gg2PbswBu6zuZSseTVgqtcp1pwl5w%3D%3D)

# The more is the size of garage the more the house cost but it seems an exceptions for 4 maybe because others factors may downgraded. This may be considered as outliers.

# # Now lets start analysing each feature and target variable.

# **SalesPrice**

# In[ ]:


labels_df.isnull().sum() # Hence no nan values in Salesprice


# In[ ]:


labels_df.shape,train_df.shape # Thus both have same no of records so no issue with Salesprice 


# In[ ]:


labels_df.describe() 


# Lets visualise this above data with Boxplot to getter a better look at it

# In[ ]:


import seaborn as sns
plt.figure(figsize=(16, 6))
sns.set(style="whitegrid")
ax = sns.boxplot(x=labels_df)


# From the above we can see that there are several outliers in sales prices.Now lets more investigate more using distribution plot
# and look deep into it.

# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(16, 6))
sns.distplot(train_df_dup['SalePrice']);


# Lets design the function we can make boxplot,probablity distribution function(pdf) so that we can get a good reproducible code

# In[ ]:


def fig_plot(data, measure):
    fig = plt.figure(figsize=(20,7))

    #Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(data)

    #Kernel Density plot
    fig1 = fig.add_subplot(121)
    sns.distplot(data, fit=norm)
    fig1.set_title(measure + ' Distribution ( mu = {:.2f} and sigma = {:.2f} )'.format(mu, sigma), loc='center')
    fig1.set_xlabel(measure)
    fig1.set_ylabel('Frequency')

    #QQ plot
    fig2 = fig.add_subplot(122)
    res = probplot(data, plot=fig2)
    fig2.set_title(measure + ' Probability Plot (skewness: {:.6f} and kurtosis: {:.6f} )'.format(data.skew(), data.kurt()), loc='center')

    plt.tight_layout()
    plt.show()


# In[ ]:


fig_plot(train_df_dup.SalePrice, 'Sales Price')


# Clearly we can see that there is skewness in pdf, so we need to fix this using transformation.

# In[ ]:


labels_df = pd.DataFrame(labels_df)


# In[ ]:


labels_df['SalePrice'].head()


# In[ ]:


# We will try the log transformation to see that can we change it to a normal distribution.


# In[ ]:



labels_df.SalePrice = np.log1p(labels_df.SalePrice)

fig_plot(labels_df.SalePrice, 'Log1P of Sales Price')


# Now we have done this I guess finding the skewness and kurtosis and skewness of all feature all at once would be better

# In[ ]:


def rstr(df, pred): 
    obs = df.shape[0]
    types = df.dtypes
    counts = df.apply(lambda x: x.count())
    uniques = df.apply(lambda x: [x.unique()])
    nulls = df.apply(lambda x: x.isnull().sum())
    distincts = df.apply(lambda x: x.unique().shape[0])
    missing_ration = (df.isnull().sum()/ obs) * 100
    skewness = df.skew()
    kurtosis = df.kurt() 
    print('Data shape:', df.shape)
    
    if pred:
        corr = df.corr()[pred]
        str = pd.concat([types, counts, distincts, nulls, missing_ration, uniques, skewness, kurtosis, corr], axis = 1, sort=False)
        corr_col = 'corr '  + pred
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing_ration', 'uniques', 'skewness', 'kurtosis', corr_col ]
    
    str.columns = cols
    dtypes = str.types.value_counts()
    print('___________________________\nData types:\n',str.types.value_counts())
    print('___________________________')
    return str


# In[ ]:


details = rstr(train_df_dup, 'SalePrice')
display(details.sort_values(by='corr SalePrice', ascending=False))


# Nulls: The data have 19 features with nulls, five of then area categorical and with more then 47% of missing ration. They are candidates to drop or use them to create another more interesting feature:<br>
# * PoolQC
# * MiscFeature
# * Alley
# * Fence
# * FireplaceQu
# 
# Features high skewed right, heavy-tailed distribution, and with high correlation to Sales Price. It is important to treat them (boxcox 1p transformation, Robustscaler, and drop some outliers):<br>
# * TotalBsmtSF
# * 1stFlrSF
# * GrLivArea<br>
# 
# Features skewed, heavy-tailed distribution, and with good correlation to Sales Price. It is important to treat them (boxcox 1p transformation, Robustscaler, and drop some outliers):<br>
# 
# 
# * LotArea
# * KitchenAbvGr
# * ScreenPorch
# * EnclosedPorch
# * MasVnrArea
# * OpenPorchSF
# * LotFrontage
# * BsmtFinSF1
# * WoodDeckSF
# * MSSubClass
# 
# <br>
# Features high skewed, heavy-tailed distribution, and with low correlation to Sales Price. Maybe we can drop these features, or just use they with other to create a new more importants feature:
# <br>
# 
# * MiscVal
# * TSsnPorch
# * LowQualFinSF
# * BsmtFinSF2
# * BsmtHalfBa
# <br>
# 
# Features low skewed, and with good to low correlation to Sales Price. Just use a Robustscaler probably reduce the few distorcions:
# 
# 
# * BsmtUnfSF
# * 2ndFlrSF
# * TotRmsAbvGrd
# * HalfBath
# * Fireplaces
# * BsmtFullBath
# * OverallQual
# * BedroomAbvGr
# * GarageArea
# * FullBath
# * GarageCars
# * OverallCond
# <br>
# 
# Transforme from Yaer Feature to Age, 2011 - Year feature, or YEAR(TODAY()) - Year Feature
# <br>
# * YearRemodAdd:
# * YearBuilt
# * GarageYrBlt
# * YrSold

# **Overall - quality**

# In[ ]:


fig_plot(train_df_dup.OverallQual, 'OverallQual')


# For area part I think we divide house into three parts :
# * HouseArea
# * Basement Area
# * Garage Area
# * Total Bath
# 

# In[ ]:


data.set_index('Id',inplace =True)


# In[ ]:


data.head()


# In[ ]:


data.isnull().sum() > 0
print(data.shape)


# In[ ]:


data['HouseArea'] = data['GrLivArea']+data['1stFlrSF'] 
+ data['2ndFlrSF']- data['LowQualFinSF']
data.drop(['GrLivArea','1stFlrSF','2ndFlrSF','LowQualFinSF'],axis = 1,inplace = True)
print(data.shape)


# In[ ]:


train_df_dup['HouseArea'] = train_df_dup['GrLivArea']+train_df_dup['1stFlrSF'] 
+ train_df_dup['2ndFlrSF']- train_df_dup['LowQualFinSF']
train_df_dup.drop(['GrLivArea','1stFlrSF','2ndFlrSF','LowQualFinSF'],axis = 1,inplace = True)
print(train_df_dup.shape)


# In[ ]:


test_df['HouseArea'] = test_df['GrLivArea']+test_df['1stFlrSF'] 
+ test_df['2ndFlrSF']- test_df['LowQualFinSF']
test_df.drop(['GrLivArea','1stFlrSF','2ndFlrSF','LowQualFinSF'],axis = 1,inplace = True)
print(test_df.shape)


# In[ ]:


train_df_dup.isnull().sum() > 0 


# In[ ]:


data.isnull().sum()


# In[ ]:


data.describe()


# In[ ]:


# Now basement
print(train_df_dup['BsmtFinSF1'].isna().sum());
print(train_df_dup['BsmtFinSF2'].isna().sum());
print(train_df_dup['BsmtUnfSF'].isna().sum());
print(test_df['BsmtFinSF1'].isna().sum());
print(test_df['BsmtFinSF2'].isna().sum());
print(test_df['BsmtUnfSF'].isna().sum());
print(data['BsmtFinSF1'].isna().sum());
print(data['BsmtFinSF2'].isna().sum());
print(data['BsmtUnfSF'].isna().sum());


# In[ ]:


test_df['BsmtFinSF1'].fillna(0,inplace =True)
test_df['BsmtFinSF2'].fillna(0,inplace =True)
test_df['BsmtUnfSF'].fillna(0,inplace = True)
print(test_df['BsmtFinSF1'].isna().sum());
print(test_df['BsmtFinSF2'].isna().sum());
print(test_df['BsmtUnfSF'].isna().sum());
data['BsmtFinSF1'].fillna(0,inplace =True)
data['BsmtFinSF2'].fillna(0,inplace =True)
data['BsmtUnfSF'].fillna(0,inplace = True)
print(data['BsmtFinSF1'].isna().sum());
print(data['BsmtFinSF2'].isna().sum());
print(data['BsmtUnfSF'].isna().sum());


# In[ ]:


data['BasementArea'] = (data['BsmtFinSF1'] ** 2 / data['TotalBsmtSF'])
+(data['BsmtFinSF2'] ** 2 / data['TotalBsmtSF'])
-(data['BsmtUnfSF'] ** 2 / data['TotalBsmtSF']);
data.drop(['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF'],axis = 1,inplace = True)
print(data.shape)


# In[ ]:


train_df_dup['BasementArea'] = (train_df_dup['BsmtFinSF1'] ** 2 / train_df['TotalBsmtSF'])
+(train_df_dup['BsmtFinSF2'] ** 2 / train_df['TotalBsmtSF'])
-(train_df_dup['BsmtUnfSF'] ** 2 / train_df['TotalBsmtSF']);
train_df_dup.drop(['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF'],axis = 1,inplace = True)
print(train_df_dup.shape)


# In[ ]:


test_df['BasementArea'] = (test_df['BsmtFinSF1'] ** 2 / test_df['TotalBsmtSF'])
+(test_df['BsmtFinSF2'] ** 2 / test_df['TotalBsmtSF'])
-(test_df['BsmtUnfSF'] ** 2 / test_df['TotalBsmtSF']);
test_df.drop(['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF'],axis = 1,inplace = True)
print(test_df.shape)


# In[ ]:


print(train_df_dup['BasementArea'].isna().sum(),
data['BasementArea'].isna().sum())


# In[ ]:


train_df_dup['BasementArea'].fillna(0,inplace = True)
test_df['BasementArea'].fillna(0,inplace =True)
data['BasementArea'].fillna(0,inplace =True)


# In[ ]:


print(train_df_dup['BasementArea'].isna().sum(),
test_df['BasementArea'].isna().sum(),
data['BasementArea'].isna().sum())


# **Lot Frontage**

# In[ ]:


data['LotFrontage'].fillna(data['LotFrontage'].median()
,inplace =True)


# In[ ]:


data['LotFrontage'].isnull().sum()


# In[ ]:


train_df_dup['LotFrontage'].fillna(train_df_dup['LotFrontage'].median()
,inplace =True)


# In[ ]:


train_df_dup['LotFrontage'].isna().sum()


# In[ ]:


test_df['LotFrontage'].fillna(test_df['LotFrontage'].median()
,inplace =True)


# 
# **MSSubClass**

# In[ ]:


train_df_dup.corr()['SalePrice']


# Since the correlation of the matrix is so low and negative so I guess its better to drop out from the Dataframe.

# In[ ]:


data.drop(['MSSubClass'],axis = 1,inplace = True)
print(data.shape)


# In[ ]:


train_df_dup.drop(['MSSubClass'],axis = 1,inplace = True)
print(train_df_dup.shape)


# In[ ]:


test_df.drop(['MSSubClass'],axis = 1,inplace = True)
print(test_df.shape)


# **MSZoning**

# In[ ]:


import seaborn as sns
sns.set(style="whitegrid")
ax = sns.barplot(x="MSZoning", y="SalePrice", data=train_df_dup)


# In[ ]:


data['MSZoning'].isna().sum()
data['MSZoning'].fillna('FV',inplace =True)
data['MSZoning'].isna().sum()


# In[ ]:


Ms = pd.get_dummies(train_df_dup['MSZoning'])
Ms1 = pd.get_dummies(test_df['MSZoning'])
Ms2 = pd.get_dummies(data['MSZoning'])


# In[ ]:


data['MSZoning'].value_counts()


# In[ ]:




train_df_dup['MSZoning'].value_counts()


# In[ ]:


train_df_dup = pd.concat([train_df_dup,Ms],axis = 1)
print(train_df_dup.shape)
test_df =  pd.concat([test_df,Ms1],axis = 1)
print(test_df.shape)


# In[ ]:


data = pd.concat([data,Ms2],axis = 1)
print(data.shape)


# **LotFrontage**

# In[ ]:


print(train_df_dup['LotFrontage'].isna().sum(),
data['LotFrontage'].isna().sum())


# **LotArea**

# In[ ]:


print(train_df_dup['LotArea'].isna().sum(),
data['LotArea'].isna().sum())


# **Street**

# We will use find and replace technique for this categorical feature.

# In[ ]:


print(train_df_dup['Street'].value_counts())
print(train_df_dup['Alley'].value_counts())
print(test_df['Street'].value_counts())
print(test_df['Alley'].value_counts())
print(data['Street'].value_counts())
print(data['Alley'].value_counts())


# In[ ]:


replace_names = {"Street" : {"Grvl" : 0,"Pave" : 1},
                 "Alley" : {"Grvl": 1 ,"Pave" : 2 , "NA" : 0}}
train_df_dup.replace(replace_names,inplace =True)
test_df.replace(replace_names,inplace =True)
data.replace(replace_names,inplace =True)
print(data['Street'].value_counts())
print(data['Alley'].value_counts())
print(train_df_dup['Street'].value_counts())
print(train_df_dup['Alley'].value_counts())
print(test_df['Street'].value_counts())
print(test_df['Alley'].value_counts())


# In[ ]:


dict(train_df_dup.groupby('LotShape')['SalePrice'].mean())


# In[ ]:


print(train_df_dup.shape,
test_df.shape,data.shape)


# In[ ]:


from scipy import stats
sns.distplot(train_df_dup['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(train_df_dup['SalePrice'], plot=plt)


# In[ ]:


train_df_dup.isnull().sum()


# In[ ]:


train_df_dup['Alley'].isna().sum()


# In[ ]:


test_df['Alley'].isna().sum()


# In[ ]:


data['Alley'].isna().sum()


# In[ ]:


train_df_dup['Alley'].value_counts()


# In[ ]:


test_df['Alley'].value_counts()


# In[ ]:


data['Alley'].value_counts()


# In[ ]:


train_df_dup.Alley.fillna(0,inplace = True)


# In[ ]:


test_df.Alley.fillna(0,inplace = True)


# In[ ]:


data.Alley.fillna(0,inplace = True)


# In[ ]:


data.Alley.value_counts()


# In[ ]:


data.isna().sum()


# In[ ]:


train_df_dup['Alley'].value_counts()


# In[ ]:


test_df['Alley'].value_counts()


# In[ ]:


display(train_df_dup.Electrical.value_counts())
display(test_df.Electrical.value_counts())

train_df_dup.Electrical.fillna('SBrKr',inplace = True)
train_df_dup.Electrical.isnull().sum()


# In[ ]:


data.Electrical.fillna('SBrKr',inplace = True)
data.Electrical.isnull().sum()


# In[ ]:


list = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']


# In[ ]:


for i in list:
    print(train_df_dup[i].value_counts())
    train_df_dup[i].fillna(train_df_dup[i].mode()[0],inplace = True)
    print(train_df_dup[i].isnull().sum())


# In[ ]:


list1 = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','TotalBsmtSF','MasVnrType'
         ,'MasVnrArea','Utilities','Exterior1st','MSZoning','Exterior2nd','SaleType']
for j in list1:
    print(test_df[j].value_counts())
    test_df[j].fillna(test_df[j].mode()[0],inplace = True)
    print(test_df[j].isnull().sum())


# In[ ]:


list2 = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','TotalBsmtSF','MasVnrType'
         ,'MasVnrArea','Utilities','Exterior1st','MSZoning','Exterior2nd','SaleType']
for j in list2:
    print(data[j].value_counts())
    data[j].fillna(data[j].mode()[0],inplace = True)
    print(data[j].isnull().sum())


# In[ ]:


data.isna().sum()


# In[ ]:


train_df_dup.isna().sum()


# In[ ]:


train_df_dup.MiscFeature.value_counts()


# In[ ]:


train_df_dup.MiscFeature.fillna('None',inplace = True)


# In[ ]:


train_df_dup.MiscFeature.value_counts()


# In[ ]:


train_df_dup.Fence.value_counts()


# In[ ]:


train_df_dup.Fence.fillna('None',inplace = True)


# In[ ]:


train_df_dup.Fence.value_counts()


# In[ ]:


train_df_dup.PoolQC.value_counts()


# In[ ]:


train_df_dup.PoolQC.fillna('None',inplace = True)


# In[ ]:


train_df_dup.PoolQC.value_counts()


# In[ ]:


l3 = ['PoolQC','Fence','MiscFeature']
for i in l3:
    test_df[i].fillna('None',inplace = True)
    print(test_df[i].isna().sum())


# In[ ]:


l4 = ['PoolQC','Fence','MiscFeature']
for i in l4:
    data[i].fillna('None',inplace = True)
    print(data[i].isna().sum())


# In[ ]:


l1 = ['FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond'] 
for j in l1:
    print(train_df_dup[j].value_counts())
    train_df_dup[j].fillna(train_df_dup[j].mode()[0],inplace = True)
    print(train_df_dup[j].isnull().sum())


# In[ ]:


l2 = ['FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','GarageCars','GarageArea'
      ,'BsmtFullBath','BsmtHalfBath','Functional','KitchenQual'] 
for j in l2:
    print(test_df[j].value_counts())
    test_df[j].fillna(test_df[j].mode()[0],inplace = True)
    print(test_df[j].isnull().sum())


# In[ ]:


l5 = ['FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','GarageCars','GarageArea'
      ,'BsmtFullBath','BsmtHalfBath','Functional','KitchenQual'] 
for j in l2:
    print(data[j].value_counts())
    data[j].fillna(data[j].mode()[0],inplace = True)
    print(data[j].isnull().sum())


# In[ ]:


train_df_dup.isnull().sum().sum()


# In[ ]:


data.isnull().sum().sum()


# In[ ]:


test_df.shape


# In[ ]:


train_df_dup.shape


# In[ ]:


data.shape


# In[ ]:




#Changing OverallCond into a categorical variable
data['OverallCond'] = data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
data['YrSold'] = data['YrSold'].astype(str)
data['MoSold'] = data['MoSold'].astype(str)


# # Label Encoding

# In[ ]:


from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'OverallCond', 
        'YrSold', 'MoSold')


# In[ ]:





# In[ ]:


for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(data[c].values)
    data[c] = lbl.transform(data[c].values)

# shape        
print('Shape all_data: {}'.format(data.shape))


# **Getting dummy categorical features
# **

# In[ ]:


data = pd.get_dummies(data)
print(data.shape)


# In[ ]:


Train = data[:1460]


# In[ ]:


Train.shape


# In[ ]:


Test = data[1460:]


# In[ ]:


Test.shape


# # Modelling

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
# We use the cross_val_score function of Sklearn. However this function has not a shuffle attribut, we add then one line of code, in order to shuffle the dataset prior to cross-validation
# 

# In[ ]:


n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(Train.values)
    rmse= np.sqrt(-cross_val_score(model, Train.values, labels_df, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# **Base Models**

# In[ ]:


labels_df.describe()


# * Lasso Regression

# This model may be very sensitive to outliers. So we need to made it more robust on them. For that we use the sklearn's Robustscaler() method on pipeline

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


# In[ ]:


lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))


# * Elastic Regresssion

# In[ ]:


ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))


# * Kernel Ridge Regression

# In[ ]:


KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)


# * Gradient Boosting Regression :

# In[ ]:


GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)


# * 
# 
#     XGBoost 

# In[ ]:


model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)


# * LightGBM :

# In[ ]:


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


# ***Base models scores***

# In[ ]:


score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(),
                                                score.std()))


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


# Stacking models

# We begin with this simple approach of averaging base models. We build a new class to extend scikit-learn with our model and also to laverage encapsulation and code reuse (inheritance)

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

# We just average four models here ENet, GBoost, KRR and lasso. Of course we could easily add more models in the mix.

# In[ ]:


averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)


# In[ ]:


model_xgb.fit(Train,labels_df)


# In[ ]:


Submission = pd.DataFrame()
Submission['LogSalePrice'] = model_xgb.predict(Test)


# In[ ]:


predictions = model_xgb.predict(Test)
predictions = np.expm1(predictions)
Submission['SalePrice'] = predictions


# In[ ]:


Submission['Id'] = test_df['Id']


# In[ ]:


Submission.head()


# In[ ]:


Submission.drop('LogSalePrice',axis = 1,inplace =True)


# In[ ]:


Submission.set_index('Id',inplace =True)


# 

# In[ ]:


Submission['SalePrice'].describe()


# In[ ]:


labels_df.SalePrice = np.log1p(labels_df.SalePrice)

fig_plot(labels_df.SalePrice, 'Log1P of Sales Price')


# In[ ]:



fig_plot(Submission.SalePrice, 'Sales Price')


# In[ ]:


Submission.head()


# In[ ]:


Output = Submission.to_csv('Output.csv')


# In[ ]:


Submission.to_csv('Output.csv')


# In[ ]:




