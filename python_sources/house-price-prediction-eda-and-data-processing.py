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
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as st
get_ipython().run_line_magic('matplotlib', 'inline')
import xgboost


# In[ ]:


with open('../input/house-prices-advanced-regression-techniques/data_description.txt','r') as f:
    info = f.read()


# In[ ]:


print(info)


# In[ ]:


df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
quantitative = [f for f in df.columns if df.dtypes[f] != 'object']
quantitative.remove('SalePrice')
quantitative.remove('Id')
qualitative = [f for f in df.columns if df.dtypes[f] == 'object']

test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


print("Quantitative Features: ",quantitative)
print()
print("Qualitative Attributes: ",qualitative)


# In[ ]:


df.head()


# In[ ]:


print(df.info())


# In[ ]:


total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[ ]:


plt.style.use('seaborn')
missing = df.isnull().sum()
missing = missing[missing>0]
missing.sort_values(inplace=True)
bars = missing.plot.bar(missing,color='aqua',edgecolor='black',linewidth=1)
plt.title("Missing Values")
plt.ylabel('Counts')
plt.xlabel('features')
plt.show()


# #### Setting up a threshold as 15%, features having more than 15% of missing data will be discarded and will not be used for trainig purposes.
# #### COLUMNS = 'Alley', 'PoolQC', 'Fence', 'MiscFeature' ,'FireplaceQu','LotFrontage'  contains inadequate data points. and exceeds the threshold for missing data percentage, hence these columns can be discardede for the training purpose.

# In[ ]:


columns_to_drop = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','Id']


# In[ ]:


df['SalePrice'].describe()


# In[ ]:



plt.figure(figsize=(10,6))
sns.distplot(df['SalePrice'],fit=st.norm)
sns.distplot(df['SalePrice'],fit=st.lognorm)
plt.show()


# The Kde distribution is positively skewed and it is evident LogNormal Distribution fits best for the Target.

# In[ ]:


print('Skewness= {:.3f}'.format(df['SalePrice'].skew()))
print('Kurtosis= {:.3f}'.format(df['SalePrice'].kurt()))


# In[ ]:


df['LogSalePrice'] = np.log1p(df['SalePrice'])
plt.figure(figsize=(10,6))
sns.distplot(df['LogSalePrice'],fit=st.norm,kde=False)
sns.distplot(df['LogSalePrice'],fit=st.lognorm,kde=False)
plt.show()


# 1. ### Relationship with Numerical Features (GrLiveArea and TotalBsmtSF)

# In[ ]:


def relation_with_numerical_feature(VarName,limit):
    data = pd.concat([df['SalePrice'],df[VarName]],axis=1)
    data.plot.scatter(x=VarName,y = 'SalePrice',ylim=(0,limit))
    plt.show()


# In[ ]:


relation_with_numerical_feature('GrLivArea',900000)


# In[ ]:


relation_with_numerical_feature("TotalBsmtSF",900000)


# ## 2.Relationship (boxplots) with qualitative features.

# In[ ]:


f = pd.melt(df, value_vars=quantitative)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)
g = g.map(sns.distplot, "value",kde_kws={'bw':1})


# 

# In[ ]:


for c in qualitative:
    df[c] = df[c].astype('category')
    if df[c].isnull().any():
        df[c] = df[c].cat.add_categories(['Missing'])
        df[c] = df[c].fillna('Missing')
        
def boxplot(x,y,**kwargs):
    sns.boxplot(x=x,y=y)
    x = plt.xticks(rotation=90)
    
f = pd.melt(df,id_vars=['SalePrice'],value_vars = qualitative)
g = sns.FacetGrid(f,col='variable',col_wrap=2, sharex=False, sharey=False, size=5)
g.map(boxplot,'value','SalePrice')
plt.show()


# #### some Features are more diversified in relation to the target(SalePrice). Categories of some features displays a very uniform pattern while there are some which resides in distinct ranges. Features like Neighbourhood, cond1, cond2, Housestyle, Garagetype, Garage Quality shows larger impact on target.
# #### feartures like PoolQc shows having a pool affects the price of the house greatly and Exterior Quality(ExterQual) shows similar behavious as of PoolQC.
# 

# ### ANOVA TEST

# In[ ]:


def anova(frame):
    anv = pd.DataFrame()
    anv['features'] = qualitative
    pvals = []
    for c in qualitative:
        samples = []
        for clas in frame[c].unique():
            s = frame[frame[c] == clas]['SalePrice'].values
            samples.append(s)
        pval = st.f_oneway(*samples)[1]
        pvals.append(pval)
        
    anv['pval'] = pvals
    return anv.sort_values('pval')

a = anova(df)
a['disparity'] = np.log(1./a['pval'].values)
sns.barplot(data=a, x='features', y='disparity')
x=plt.xticks(rotation=90)


# ### Here is quick estimation of influence of categorical variable on SalePrice. For each variable SalePrices are partitioned to distinct sets based on category values. Then check with ANOVA test if sets have similar distributions. If variable has minor impact then set means should be equal. Decreasing pval is sign of increasing diversity in partitions.
# ### The same results can also be seen from box plots of individual features.
# 

# ## Handling Missing Values

# In[ ]:


train = df.drop(columns_to_drop,axis=1)
test = test_data.drop(columns_to_drop,axis=1)


# In[ ]:


sns.heatmap(train.isnull(),cbar=False)


# #### Replacing all nan values with mode of that variable

# In[ ]:


def fill_nan_train(feature):
    if train[feature].isnull().sum()>0:
        if feature in qualitative:
            train[feature].fillna(train[feature].mode()[0],inplace=True)
        elif feature in quantitative:
            train[feature].fillna(train[feature].mean(),inplace=True)


# In[ ]:


columns_with_nan_values = ['LotFrontage','GarageFinish','GarageType','GarageCond','GarageQual','GarageYrBlt','BsmtFinType2','BsmtExposure','BsmtFinType1','BsmtCond',
                          'BsmtQual','MasVnrType','Electrical','MasVnrArea','MSZoning','BsmtFullBath','Utilities','BsmtHalfBath','Functional','TotalBsmtSF','GarageArea','BsmtFinSF2','BsmtUnfSF','SaleType','Exterior2nd','Exterior1st','KitchenQual','GarageCars','BsmtFinSF1']


# In[ ]:


for col in columns_with_nan_values:
    fill_nan_train(col)


# In[ ]:


train.isnull().sum().sort_values(ascending=False)


# In[ ]:


print(qualitative,len(qualitative),quantitative,len(quantitative),sep='\n')


# ### Missing Values in Test Data

# In[ ]:


sns.heatmap(test.isnull(),cbar=False,cmap = 'ocean')


# In[ ]:


def fill_nan_test(feature):
    if test[feature].isnull().sum()>0:
        if feature in qualitative:
            test[feature].fillna(test[feature].mode()[0],inplace=True)
        elif feature in quantitative:
            test[feature].fillna(test[feature].mean(),inplace=True)


# In[ ]:


for col in columns_with_nan_values:
    fill_nan_test(col)


# In[ ]:


print(test.isnull().sum().sort_values(ascending=False).head(12))


# #### In test data, there are occurences of categories of categorical data which are not present in train data. So inorder to prevent any misprediction/error due to this, so, before converting our categorical variables to One hot encodings, will use the entire data i.e. test + train inorder to obtain a Label for that missing category in our train data as well. 

# In[ ]:


print(test.shape)
print(train.shape)


# In[ ]:


for i in columns_to_drop:
    if i in qualitative:
        qualitative.remove(i)
    elif i in quantitative:
        quantitative.remove(i)


# In[ ]:


df_complete = pd.concat([train,test],axis=0)
print('train_shape',train.shape)
print('test_shape',test.shape)
print('df_complete shape',df_complete.shape)


# ### One hot encodings for Categorical Features using df_complete

# In[ ]:


final_df = df_complete.copy()


# In[ ]:


def category_onehot(features):
    new_df = df_complete
    num = 0
    print(new_df.shape)
    for col in features:
        onehot = pd.get_dummies(new_df[col],drop_first=True)
        new_df.drop(col,axis=1,inplace=True) #drop entire column
        new_df = pd.concat([new_df,onehot],axis=1)
        num += onehot.shape[1]
        print(col)
    print(new_df.shape)
    print(num)
    return new_df
    


# In[ ]:


new_df = category_onehot(qualitative)


# In[ ]:


new_df = new_df.loc[:,~new_df.columns.duplicated()]


# In[ ]:


new_df.shape


# #### Extract test and train back from the complete df. train shape --> 1460 
# 

# In[ ]:


X_train = new_df[:1460]
X_test = new_df[1460:]
Y_train = X_train[['SalePrice','LogSalePrice']]


# In[ ]:


X_train = X_train.drop(['SalePrice','LogSalePrice'],axis=1)
X_test = X_test.drop(['SalePrice','LogSalePrice'],axis=1)


# In[ ]:


Y_train1 = Y_train['LogSalePrice']


# In[ ]:


print('X_train',X_train.shape)
print('X_test',X_test.shape)
print('Y_train',Y_train.shape)


# In[ ]:


classifier = xgboost.XGBRegressor()
classifier.fit(X_train,Y_train1)


# In[ ]:


result = np.exp(classifier.predict(X_test))-1


# In[ ]:


id_column = test_data['Id']
result = result.reshape((-1,1))
result = pd.DataFrame(result,columns=['SalePrice'])


# In[ ]:


prediction = pd.concat([id_column,result],axis=1)


# In[ ]:


#prediction.to_csv('submisison.csv',index=False,header=True)


# In[ ]:





# In[ ]:




