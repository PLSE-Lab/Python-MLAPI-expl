#!/usr/bin/env python
# coding: utf-8

# Home values are influenced by many factors. Basically, there are two major aspects:
# 
#  1. The environmental information, including location, local economy, school district, air quality, etc.
#  2. The characteristics information of the property, such as lot size, house size and age, the number of rooms, heating / AC systems, garage, and so on.
# 
# When people consider buying homes, usually the location has been constrained to a certain area such as not too far from the workplace. 
# 
# **With location factor pretty much fixed, the property characteristics information weights more in the home prices**. 
# 
# There are many factors describing the condition of a house, and they do not weigh equally in determining the home value. I present a feature selection process to examine the key features affecting their values.
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# **Combining both the datasets**

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'png' #set 'png' here when working on notebook")
warnings.filterwarnings('ignore') 


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# **Combining both the datasets**

# In[ ]:


df = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']), ignore_index=True)


# **Data Preprocessing**

# **Eliminating samples or features with missing values**

# In[ ]:


#delete the column without having to reassign df you can do:
df.drop('Alley', axis=1, inplace=True)
df.drop('PoolQC', axis=1, inplace=True)
df.drop('Fence', axis=1, inplace=True)
df.drop('MiscFeature', axis=1, inplace=True)


# The rest of the data processing is from the kernel by Boris Klyus

# In[ ]:


df.loc[df.MasVnrType.isnull(), 'MasVnrType'] = 'None' # no good
df.loc[df.MasVnrType == 'None', 'MasVnrArea'] = 0
df.loc[df.LotFrontage.isnull(), 'LotFrontage'] = df.LotFrontage.median()
df.loc[df.LotArea.isnull(), 'MasVnrType'] = 0
df.loc[df.BsmtQual.isnull(), 'BsmtQual'] = 'NoBsmt'
df.loc[df.BsmtCond.isnull(), 'BsmtCond'] = 'NoBsmt'
df.loc[df.BsmtExposure.isnull(), 'BsmtExposure'] = 'NoBsmt'
df.loc[df.BsmtFinType1.isnull(), 'BsmtFinType1'] = 'NoBsmt'
df.loc[df.BsmtFinType2.isnull(), 'BsmtFinType2'] = 'NoBsmt'
df.loc[df.BsmtFinType1=='NoBsmt', 'BsmtFinSF1'] = 0
df.loc[df.BsmtFinType2=='NoBsmt', 'BsmtFinSF2'] = 0
df.loc[df.BsmtFinSF1.isnull(), 'BsmtFinSF1'] = df.BsmtFinSF1.median()
df.loc[df.BsmtQual=='NoBsmt', 'BsmtUnfSF'] = 0
df.loc[df.BsmtUnfSF.isnull(), 'BsmtUnfSF'] = df.BsmtUnfSF.median()
df.loc[df.BsmtQual=='NoBsmt', 'TotalBsmtSF'] = 0
df.loc[df.FireplaceQu.isnull(), 'FireplaceQu'] = 'NoFireplace'
df.loc[df.GarageType.isnull(), 'GarageType'] = 'NoGarage'
df.loc[df.GarageFinish.isnull(), 'GarageFinish'] = 'NoGarage'
df.loc[df.GarageQual.isnull(), 'GarageQual'] = 'NoGarage'
df.loc[df.GarageCond.isnull(), 'GarageCond'] = 'NoGarage'
df.loc[df.BsmtFullBath.isnull(), 'BsmtFullBath'] = 0
df.loc[df.BsmtHalfBath.isnull(), 'BsmtHalfBath'] = 0
df.loc[df.KitchenQual.isnull(), 'KitchenQual'] = 'TA'
df.loc[df.MSZoning.isnull(), 'MSZoning'] = 'RL'
df.loc[df.Utilities.isnull(), 'Utilities'] = 'AllPub'
df.loc[df.Exterior1st.isnull(), 'Exterior1st'] = 'VinylSd'
df.loc[df.Exterior2nd.isnull(), 'Exterior2nd'] = 'VinylSd'
df.loc[df.Functional.isnull(), 'Functional'] = 'Typ'
df.loc[df.SaleCondition.isnull(), 'SaleCondition'] = 'Normal'
df.loc[df.SaleCondition.isnull(), 'SaleType'] = 'WD'
df.loc[df['Electrical'].isnull(), 'Electrical'] = 'SBrkr'
df.loc[df['SaleType'].isnull(), 'SaleType'] = 'NoSale'
#GarageYrBlt
df.loc[df.GarageYrBlt.isnull(), 'GarageYrBlt'] = df.GarageYrBlt.median()
# only one is null and it has type Detchd
df.loc[df['GarageArea'].isnull(), 'GarageArea'] = df.loc[df['GarageType']=='Detchd', 'GarageArea'].mean()
df.loc[df['GarageCars'].isnull(), 'GarageCars'] = df.loc[df['GarageType']=='Detchd', 'GarageCars'].median()


# **Handling categorical data**

# In[ ]:


size_mapping = {'Y': 1,'N': 0}
df['CentralAir'] = df['CentralAir'].map(size_mapping)


# In[ ]:


df = pd.get_dummies(df)


# In[ ]:


#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])


# In[ ]:


#creating matrices for sklearn:
X_train = df[:train.shape[0]]
X_test = df[train.shape[0]:]
y = train.SalePrice


# **Partitioning a dataset in training and test sets**

# In[ ]:


from sklearn.cross_validation import train_test_split
X = df[:train.shape[0]].values
y = train.SalePrice.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# **Bringing features onto the same scale**

# In[ ]:


from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)


# The issue is that after your scaling step, the labels are float-valued, which is not a valid label-type; you convert to int or str for the y_train and y_test to work

# In[ ]:





# In[ ]:


y_train = y_train.astype(int)
y_train

y_test = y_test.astype(int)


# **Selecting meaningful features**

# In[ ]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l1', C=0.1)
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))


# In[ ]:


lr.intercept_


# In[ ]:


lr.coef_


# In[ ]:


fig = plt.figure()
ax = plt.subplot(111)
    
colors = ['blue', 'green', 'red', 'cyan', 
         'magenta', 'yellow', 'black', 
          'pink', 'lightgreen', 'lightblue', 
          'gray', 'indigo', 'orange']

weights, params = [], []
for c in np.arange(-4, 6):
    lr = LogisticRegression(penalty='l1', C=10**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)

for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column],
             label=df.columns[column+1],
             color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', 
          bbox_to_anchor=(1.38, 1.03),
          ncol=1, fancybox=True)
plt.show()


# **The resulting plot provides us with further insights about the behavior of L1 regularization.**

# **Assessing feature importance with random forests**
# 
# Using a random forest, we can measure feature importance as the averaged impurity decrease computed from all decision trees in the forest without making any assumptions whether our data is linearly separable or not.
# 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train_std, y_train)

feature_imp = pd.DataFrame(model.feature_importances_, index=df.columns, columns=["importance"])
feat_imp_20 = feature_imp.sort_values("importance", ascending=False).head(20).index
feat_imp_20


# **Feature selector that removes all low-variance features.**
# 
# This feature selection algorithm looks only at the features (X), not the desired outputs (y), and can thus be used for unsupervised learning.
# 
# 

# In[ ]:


from sklearn.feature_selection import VarianceThreshold, f_regression, SelectKBest

#Find all features with more than 90% variance in values.
threshold = 0.90
vt = VarianceThreshold().fit(X_train_std)

# Find feature names
feat_var_threshold = df.columns[vt.variances_ > threshold * (1-threshold)]
# select the top 20 

feat_var_threshold[0:20]


# **Univariate feature selection**
# 
# Univariate feature selection works by selecting the best features based on univariate statistical tests. It can be seen as a preprocessing step to an estimator. Scikit-learn exposes feature selection routines as objects that implement the transform method.

# In[ ]:


X_scored = SelectKBest(score_func=f_regression, k='all').fit(X_train_std, y_train)
feature_scoring = pd.DataFrame({
        'feature': df.columns,
        'score': X_scored.scores_
    })

feat_scored_20 = feature_scoring.sort_values('score', ascending=False).head(20)['feature'].values
feat_scored_20


# **Recursive Feature Elimination**

# In[ ]:


#Select 20 features from using recursive feature elimination (RFE) with logistic regression model.
from sklearn.feature_selection import RFE
rfe = RFE(LogisticRegression(), 20)
rfe.fit(X_train_std, y_train)

feature_rfe_scoring = pd.DataFrame({
        'feature': df.columns,
        'score': rfe.ranking_
    })

feat_rfe_20 = feature_rfe_scoring[feature_rfe_scoring['score'] == 1]['feature'].values
feat_rfe_20


# **Final feature selection**

# In[ ]:


features = np.hstack([
        feat_var_threshold[0:20], 
        feat_imp_20,
        feat_scored_20,
        feat_rfe_20
    ])

features = np.unique(features)
print('Final features set:\n')
for f in features:
    print("\t-{}".format(f))


# After feature selection, it looks like my hypothesis is true that the property characteristics information weights more location in the home prices.
# 
# What do you all think ? Please leave a comment for [me][1]. 
# 
# 
#   [1]: http://piushvaish.com
# 
# Thanks
# Piush
